import os

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLMWithMetrics
from filters.mwfilter_lightning.mwfilter_base_lm import MWFilterBaseLMWithMetricsAE
from models import CorrectionNet
from mwlab import TouchstoneDataset, TouchstoneDatasetAnalyzer, TouchstoneLDataModule
from mwlab.nn import MinMaxScaler
from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample

from filters import CMTheoreticalDatasetGeneratorSamplers, SamplerTypes, MWFilter, CouplingMatrix
from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift, CMTheoreticalDatasetGenerator
import models
import configs
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger




def plot_distribution(ds: TouchstoneDataset, num_params: int, batch: int = 6):
    analyzer = TouchstoneDatasetAnalyzer(ds)
    varying = analyzer.get_varying_keys()
    r = int((num_params / batch) + num_params % batch)
    for i in range(r):
        analyzer.plot_param_distributions(varying[batch * i:batch * (i + 1)])


def create_origin_filter(path_orig_filter: str, f_start=None, f_stop=None, f_unit=None, resample_scale=301):
    tds = TouchstoneDataset(source=path_orig_filter)
    origin_filter = MWFilter.from_touchstone_dataset_item(tds[0])
    f0 = origin_filter.f0
    bw = origin_filter.bw
    if f_start is None:
        f_start = f0 - 3 * bw
    if f_stop is None:
        f_stop = f0 + 3 * bw
    if f_unit is None:
        f_unit = "MHz"
    y_transform = TComposite([
        S_Crop(f_start=f_start, f_stop=f_stop, unit=f_unit),
        S_Resample(resample_scale)
    ])
    tds_transformed = TouchstoneDataset(source=path_orig_filter, s_tf=y_transform)
    origin_filter = MWFilter.from_touchstone_dataset_item(tds_transformed[0])
    return origin_filter


def create_sampler(orig_filter: MWFilter, sampler_type: SamplerTypes, sampler_configs:dict, with_one_param: bool=False):
    samplers_all_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                    samplers_type=sampler_type(
                                                                                        one_param=False),
                                                                                    **sampler_configs)
    samplers_all_params_shuffle_cms_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_all_params.cms.shuffle(ratio=1, dim=1),
        pss=samplers_all_params.pss)
    samplers_all_params_shuffle_pss_cols = CMTheoreticalDatasetGeneratorSamplers(cms=samplers_all_params.cms,
                                                                                     pss=samplers_all_params.pss.shuffle(
                                                                                         ratio=1, dim=1))
    samplers_all_params_flip_sign_cms_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_all_params.cms.flip_signs(ratio=1, dim=0),
        pss=samplers_all_params.pss
    )
    samplers_all_params_shuffle_all_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_all_params.cms.shuffle(ratio=1, dim=1),
        pss=samplers_all_params.pss.shuffle(ratio=1, dim=1)
    )

    sampler_configs["samplers_size"] = int(sampler_configs["samplers_size"] / 100)
    samplers_with_one_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                         samplers_type=sampler_type(
                                                                                             one_param=True),
                                                                                         **sampler_configs)
    if with_one_param:
        total_samplers = CMTheoreticalDatasetGeneratorSamplers.concat(
            (samplers_all_params, samplers_with_one_params)
        )
        total_samplers.cms._type = sampler_type
        total_samplers.pss._type = sampler_type
    else:
        # total_samplers = CMTheoreticalDatasetGeneratorSamplers.concat(
        #     (samplers_all_params, samplers_all_params_shuffle_pss_cols)
        # )
        total_samplers = samplers_all_params
        total_samplers.cms._type = sampler_type
        total_samplers.pss._type = sampler_type
    if torch.isnan(total_samplers.cms.space).any() or torch.isinf(total_samplers.cms.space).any():
        raise ValueError("⚠️ (cms) Input to model contains NaN or Inf")
    if torch.isnan(total_samplers.pss.space).any() or torch.isinf(total_samplers.pss.space).any():
        raise ValueError("⚠️ (pss) Input to model contains NaN or Inf")
    return total_samplers


def train_model(model: nn.Module, dm: TouchstoneLDataModule, trainer: L.Trainer,
                loss_fn,
                optimizer_cfg:dict={"name": "AdamW", "lr": 0.0005370623202982373, "weight_decay": 1e-5},
                scheduler_cfg:dict={"name": "StepLR", "step_size": 21, "gamma": 0.01}):
    lit_model = MWFilterBaseLMWithMetrics(
        model=model,  # Наша нейросетевая модель
        swap_xy=True,
        scaler_in=dm.scaler_in,  # Скейлер для входных данных
        scaler_out=dm.scaler_out,  # Скейлер для выходных данных
        codec=dm.codec,  # Кодек для преобразования данных
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        loss_fn=loss_fn
    )
    trainer.fit(lit_model, dm)
    return lit_model

def check_metrics(trainer: L.Trainer, model: L.LightningModule, dm: TouchstoneLDataModule):
    print("Train loader")
    train_metrics = trainer.test(model, dataloaders=dm.train_dataloader())
    print("Validation loader")
    val_metrics = trainer.test(model, dataloaders=dm.val_dataloader())
    print("Test loader")
    test_metrics = trainer.test(model, dataloaders=dm.test_dataloader())
    return {"train": train_metrics, "val": val_metrics, "test": test_metrics}


class MySafeCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")

        # Только если оба есть
        if train_loss is not None and val_loss is not None:
           if train_loss < val_loss:
                # Не сохраняем модель — early overfitting
                return

        super().on_validation_end(trainer, pl_module)


class WorkModel:
    CHECKPOINT_DIRPATH = "saved_models/" + configs.FILTER_NAME
    BEST_MODEL_FILENAME_SUFFIX = "-batch_size={batch_size}-base_dataset_size={base_dataset_size}-sampler={sampler_type}"
    def __init__(self, ds_path, sampler_configs:dict, sampler_type=SamplerTypes.SAMPLER_SOBOL):
        L.seed_everything(0)
        print("Создаем фильтр")
        self.orig_filter = create_origin_filter(configs.ENV_ORIGIN_DATA_PATH, resample_scale=301)
        print("Создаем сэмплеры")
        self.sampler_configs = sampler_configs
        self.ds_size = sampler_configs["samplers_size"]
        self.ds_path = ds_path
        self.samplers = create_sampler(self.orig_filter, sampler_type, sampler_configs=sampler_configs)
        self.ds_gen = CMTheoreticalDatasetGenerator(
            path_to_save_dataset=os.path.join(ds_path, self.samplers.cms.type.name, f"{len(self.samplers.cms)}"),
            backend_type='ram',
            orig_filter=self.orig_filter,
            filename="Dataset",
        )
        self.ds_gen.generate(self.samplers)

        self.ds = TouchstoneDataset(source=self.ds_gen.backend, in_memory=True)

        self.trainer = self._configure_trainer()
        # self.codec = MWFilterTouchstoneCodec.from_dataset(ds=self.ds,
        #                                          keys_for_analysis=[f"m_{r}_{c}" for r, c in self.orig_filter.coupling_matrix.links])
        self.model_name = None
        self.meta = None
        self.codec = None
        self.model = None
        self.dm = None

    @staticmethod
    def get_model(name: str = "resnet_with_correction", **kwargs):
        if name == "resnet":
            resnet = models.ResNet1DFlexible(
                in_channels=kwargs["in_channels"],
                out_channels=kwargs["out_channels"],
                num_blocks=[1, 4, 3, 5],
                layer_channels=[64, 64, 128, 256],
                first_conv_kernel=8,
                first_conv_channels=64,
                first_maxpool_kernel=2,
                activation_in='sigmoid',
                activation_block='swish',
                use_se=False,
                se_reduction=1
            )
            # resnet = models.ResNet1DFlexible(
            #     in_channels=kwargs["in_channels"],
            #     out_channels=kwargs["out_channels"],
            #     num_blocks=[4, 6, 4, 4],
            #     layer_channels=[128, 256, 64, 128],
            #     first_conv_kernel=9,
            #     first_conv_channels=128,
            #     first_maxpool_kernel=9,
            #     activation_in='gelu',
            #     activation_block='mish',
            #     # use_se=False,
            #     # se_reduction=8
            # )
            # resnet = models.ResNet1DFlexible(
            #     in_channels=kwargs["in_channels"],
            #     out_channels=kwargs["out_channels"],
            #     num_blocks=[4, 6, 4, 4],
            #     layer_channels=[128, 256, 64, 128],
            #     first_conv_kernel=9,
            #     first_conv_channels=128,
            #     first_maxpool_kernel=9,
            #     activation_in='gelu',
            #     activation_block='mish',
            #     # use_se=False,
            #     # se_reduction=1
            # )
            resnet = models.ResNet1DFlexible(
                in_channels=kwargs["in_channels"],
                out_channels=kwargs["out_channels"],
                first_conv_channels=512,
                first_conv_kernel=7,
                first_maxpool_kernel=11,
                block_kernel_size=7,
                layer_channels=[512, 512, 512, 256],
                num_blocks=[8, 8, 5, 8],
                activation_in='leaky_relu',
                activation_block='rrelu',
                block_type='bottleneck'
            )
            return resnet
        elif name == "resnet_2d":
            main = models.ResNet2DFlexible(
                freq_vector=kwargs["freq_vector"],
                in_channels=kwargs["in_channels"],
                out_channels=kwargs["out_channels"],
                activation_in='sigmoid',
                activation_block='swish',
                num_blocks=[1, 4, 3, 5],
                layer_channels=[64, 64, 128, 256],
                first_conv_kernel=8,
                first_conv_channels=64,
                first_maxpool_kernel=3,
                use_se=True,
                se_reduction=4
            )
            return main
        elif name == "resnet_with_correction":
            # main = models.ResNet1DFlexible(
            #     in_channels=kwargs["in_channels"],
            #     out_channels=kwargs["out_channels"],
            #     num_blocks=[3, 7, 7, 4],
            #     layer_channels=[512, 1024, 16, 128],
            #     first_conv_kernel=9,
            #     first_conv_channels=512,
            #     first_maxpool_kernel=9,
            #     activation_in='gelu',
            #     activation_block='mish',
            # )
            # mlp = models.CorrectionMLP(
            #     input_dim=kwargs["out_channels"],
            #     output_dim=kwargs["out_channels"],
            #     hidden_dims=[8, 512, 8],
            #     activation_fun='mish'
            # )

            # main = models.ResNet1DFlexible(
            #     in_channels=kwargs["in_channels"],
            #     out_channels=kwargs["out_channels"],
            #     num_blocks=[4, 6, 4, 4],
            #     layer_channels=[128, 256, 64, 128],
            #     first_conv_kernel=9,
            #     first_conv_channels=128,
            #     first_maxpool_kernel=9,
            #     activation_in='gelu',
            #     activation_block='mish',
            #     # block_kernel_size=5,
            #     # use_se=False,
            #     # se_reduction=1
            # )
            # mlp = models.CorrectionMLP(
            #     input_dim=kwargs["out_channels"],
            #     output_dim=kwargs["out_channels"],
            #     hidden_dims=[2048, 4096, 256],
            #     activation_fun='gelu'
            # )

            # main = models.ResNet1DFlexible(
            #     in_channels=kwargs["in_channels"],
            #     out_channels=kwargs["out_channels"],
            #     num_blocks=[7, 7, 1, 3],
            #     layer_channels=[512, 64, 64, 32],
            #     block_kernel_size=7,
            #     block_type='basic',
            #     first_conv_kernel=11,
            #     first_conv_channels=1024,
            #     first_maxpool_kernel=3,
            #     activation_in='leaky_relu',
            #     activation_block='leaky_relu',
            #     # use_se=False,
            #     # se_reduction=1
            # )
            # mlp = models.CorrectionMLP(
            #     input_dim=kwargs["out_channels"],
            #     output_dim=kwargs["out_channels"],
            #     hidden_dims=[256, 4096, 512],
            #     activation_fun='soft_sign'
            # )
            main = WorkModel.get_model('resnet', **kwargs)
            mlp = models.CorrectionMLP(
                input_dim=kwargs["out_channels"],
                output_dim=kwargs["out_channels"],
                hidden_dims=[2048, 2048, 128],
                activation_fun='sin'
            )
            model = models.ModelWithCorrection(
                main_model=main,
                correction_model=mlp,
            )
            return model
        elif name == "resnet_with_wide_correction":
            residual_correction_model = kwargs["main_model"]
            wide_correction_model = models.ModelWithCorrectionAndSparameters(
                main_model=residual_correction_model,
                correction_model=CorrectionNet(s_shape=(8, 301), m_dim=kwargs["out_channels"])
            )
            return wide_correction_model
        elif name == "simple_opt":
            simple_opt_model = models.Simple_Opt_3(**kwargs)
            return simple_opt_model
        elif name == "birnn":
            birnn = models.BiRNN(**kwargs)
            return birnn
        else:
            raise ValueError(f"Unknown model name: {name}")

    def _configure_trainer(self):
        stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_mae", patience=31, mode="min", min_delta=0.00001)
        checkpoint = MySafeCheckpoint(monitor="val_mae",
                                      dirpath="saved_models/" + configs.FILTER_NAME,
                                      filename="best-{epoch}-{train_loss:.5f}-{val_loss:.5f}-{val_r2:.5f}-{val_mse:.5f}-"
                                                                  "{val_mae:.5f}" + self.BEST_MODEL_FILENAME_SUFFIX.format(
                                                             batch_size=configs.BATCH_SIZE,
                                                             base_dataset_size=self.ds_size,
                                                             sampler_type=self.samplers.cms.type
                                                         ),
                                      mode="min",
                                      save_top_k=1,  # Сохраняем только одну лучшую
                                      save_weights_only=False,
                                      # Сохранять всю модель (включая структуру)
                                      verbose=False  # Отключаем логирование сохранения)
                                      )
        trainer = L.Trainer(
            deterministic=True,
            max_epochs=30,  # Максимальное количество эпох обучения
            accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
            log_every_n_steps=100,  # Частота логирования в процессе обучения
            callbacks=[stoping, checkpoint]
        )
        return trainer

    def setup(self, model_name: str, model_cfg: dict, dm_codec: MWFilterTouchstoneCodec):
        self.model_name = model_name
        self.model = self.get_model(model_name, **model_cfg)
        # self.model = self.inference("saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=29-train_loss=0.04166-val_loss=0.04450-val_r2=0.92560-val_mse=0.00588-val_mae=0.03862-batch_size=32-base_dataset_size=1500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
        # self.model = models.ModelWithCorrectionAndSparameters(
        #     main_model=self.model,
        #     correction_model=CorrectionNet(s_shape=(8, 301), m_dim=len(dm_codec.x_keys))
        # )
        print(dm_codec)
        print("Каналы Y:", dm_codec.y_channels)
        print("Каналы X:", dm_codec.x_keys)
        print("Количество каналов:", len(dm_codec.y_channels))
        self.codec = dm_codec
        self.dm = TouchstoneLDataModule(
            source=self.ds_gen.backend,  # Путь к датасету
            codec=dm_codec,  # Кодек для преобразования TouchstoneData → (x, y)
            batch_size=configs.BATCH_SIZE,  # Размер батча
            val_ratio=0.2,  # Доля валидационного набора
            test_ratio=0.05,  # Доля тестового набора
            cache_size=0,
            scaler_in=MinMaxScaler(dim=(0, 2), feature_range=(0, 1)),  # Скейлер для входных данных
            scaler_out=MinMaxScaler(dim=0, feature_range=(-0.5, 0.5)),  # Скейлер для выходных данных
            swap_xy=True,
            num_workers=0,
            # Параметры базового датасета:
            base_ds_kwargs={
                "in_memory": True
            }
        )
        self.dm.setup("fit")
        # Печатаем размеры полученных наборов
        print(f"Размер тренировочного набора: {len(self.dm.train_ds)}")
        print(f"Размер валидационного набора: {len(self.dm.val_ds)}")
        print(f"Размер тестового набора: {len(self.dm.test_ds)}")

        # Возьмем для примера первый touchstone-файл из тестового набора данных
        test_tds = self.dm.get_dataset(split="test", meta=True)
        # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
        _, _, self.meta = test_tds[0]  # Используем первый файл набора данных]


    def train(self, optimizer_cfg: dict, scheduler_cfg: dict, loss_fn):
        tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=f"{self.model_name}")
        csv_logger = CSVLogger(save_dir="lightning_logs", name=f"{self.model_name}_csv")
        self.trainer.loggers = [tb_logger, csv_logger]
        lit_model = MWFilterBaseLMWithMetrics(
            model=self.model,  # Наша нейросетевая модель
            swap_xy=True,
            scaler_in=self.dm.scaler_in,  # Скейлер для входных данных
            scaler_out=self.dm.scaler_out,  # Скейлер для выходных данных
            codec=self.dm.codec,  # Кодек для преобразования данных
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            loss_fn=loss_fn
        )
        self.trainer.fit(lit_model, self.dm)
        # lit_model = train_model(model=self.model, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, dm=self.dm,
        #             trainer=self.trainer, loss_fn=loss_fn)
        print(f"Лучшая модель сохранена в: {self.trainer.checkpoint_callback.best_model_path}")
        return lit_model

    def inference(self, path_to_ckpt: str):
        inference_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
            checkpoint_path=path_to_ckpt,
            model=self.model
        )
        return inference_model


class AEWorkModel(WorkModel):
    def __init__(self, ds_path, sampler_configs, sampler_type=SamplerTypes.SAMPLER_SOBOL):
        super().__init__(ds_path, sampler_configs, sampler_type)

    @staticmethod
    def get_model(name: str = "cae", **kwargs):
        if name=="cae":
            model = models.ConvAE(**kwargs)
            return model
        elif name=="mlp":
            pass
        elif name=="rae":
            model = models.RNNAutoencoder(**kwargs)
            return model
        elif name=="imp_cae":
            model = models.ConvAE_ShuffleDet(**kwargs)
            return model
        else:
            ValueError(f"Unknown model name: {name}")

    def setup(self, model_name: str, model_cfg: dict, dm_codec: MWFilterTouchstoneCodec):
        self.model_name = model_name
        self.model = self.get_model(model_name, **model_cfg)
        # self.model = self.inference("saved_models\\EAMU4-KuIMUXT3-BPFC1\\best-epoch=29-train_loss=0.04166-val_loss=0.04450-val_r2=0.92560-val_mse=0.00588-val_mae=0.03862-batch_size=32-base_dataset_size=1500000-sampler=SamplerTypes.SAMPLER_SOBOL.ckpt")
        # self.model = models.ModelWithCorrectionAndSparameters(
        #     main_model=self.model,
        #     correction_model=CorrectionNet(s_shape=(8, 301), m_dim=len(dm_codec.x_keys))
        # )
        print(dm_codec)
        print("Каналы Y:", dm_codec.y_channels)
        print("Каналы X:", dm_codec.x_keys)
        print("Количество каналов:", len(dm_codec.y_channels))
        self.codec = dm_codec
        self.dm = TouchstoneLDataModule(
            source=self.ds_gen.backend,  # Путь к датасету
            codec=dm_codec,  # Кодек для преобразования TouchstoneData → (x, y)
            batch_size=configs.BATCH_SIZE,  # Размер батча
            val_ratio=0.2,  # Доля валидационного набора
            test_ratio=0.05,  # Доля тестового набора
            cache_size=0,
            scaler_in=MinMaxScaler(dim=(0, 2), feature_range=(0, 1)),  # Скейлер для входных данных
            scaler_out=MinMaxScaler(dim=0, feature_range=(-0.5, 0.5)),  # Скейлер для выходных данных
            swap_xy=True,
            num_workers=0,
            # Параметры базового датасета:
            base_ds_kwargs={
                "in_memory": True
            }
        )
        self.dm.setup("fit")
        # Печатаем размеры полученных наборов
        print(f"Размер тренировочного набора: {len(self.dm.train_ds)}")
        print(f"Размер валидационного набора: {len(self.dm.val_ds)}")
        print(f"Размер тестового набора: {len(self.dm.test_ds)}")

        # Возьмем для примера первый touchstone-файл из тестового набора данных
        test_tds = self.dm.get_dataset(split="test", meta=True)
        # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
        _, _, self.meta = test_tds[0]  # Используем первый файл набора данных]

    def train(self, optimizer_cfg: dict, scheduler_cfg: dict, loss_fn):
        tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=f"{self.model_name}")
        csv_logger = CSVLogger(save_dir="lightning_logs", name=f"{self.model_name}_csv")
        self.trainer.loggers = [tb_logger, csv_logger]
        lit_model = MWFilterBaseLMWithMetricsAE(
            meta=self.meta,
            model=self.model,  # Наша нейросетевая модель
            swap_xy=True,
            scaler_in=self.dm.scaler_in,  # Скейлер для входных данных
            scaler_out=self.dm.scaler_out,  # Скейлер для выходных данных
            codec=self.dm.codec,  # Кодек для преобразования данных
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            loss_fn=loss_fn
        )
        self.trainer.fit(lit_model, self.dm)
        print(f"Лучшая модель сохранена в: {self.trainer.checkpoint_callback.best_model_path}")
        return lit_model

    def inference(self, path_to_ckpt: str):
        inference_model = MWFilterBaseLMWithMetricsAE.load_from_checkpoint(
            checkpoint_path=path_to_ckpt,
            model=self.model,
            meta=self.meta
        )
        return inference_model
