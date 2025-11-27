import os

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLMWithMetrics
from models import CorrectionNet
from mwlab import TouchstoneDataset, TouchstoneDatasetAnalyzer, TouchstoneLDataModule, TouchstoneCodec
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
        # f_start = f0 - 1 * bw + bw/4
        f_start = configs.F_START_MHZ
    if f_stop is None:
        # f_stop = f0 + 1 * bw - bw/4
        f_stop = configs.F_STOP_MHZ
    if f_unit is None:
        f_unit = "MHz"
    print(f"f0={f0}, bw={bw}, f_start={f_start}, f_stop={f_stop}")
    y_transform = TComposite([
        S_Crop(f_start=f_start, f_stop=f_stop, unit=f_unit),
        S_Resample(resample_scale)
    ])
    tds_transformed = TouchstoneDataset(source=path_orig_filter, s_tf=y_transform)
    origin_filter = MWFilter.from_touchstone_dataset_item(tds_transformed[0])
    return origin_filter


def create_sampler(orig_filter: MWFilter, sampler_type: SamplerTypes, with_one_param: bool=False, dataset_size=configs.BASE_DATASET_SIZE):
    sampler_configs = {
        "pss_origin": PSShift(a11=0.0, a22=0.0, b11=0, b22=0),
        "pss_shifts_delta": PSShift(a11=0.1, a22=0.1, b11=0.1, b22=0.1),
        # "cm_shifts_delta": CMShifts(self_coupling=1.8, mainline_coupling=0.3, cross_coupling=9e-2, parasitic_coupling=5e-3),
        # "cm_shifts_delta": CMShifts(self_coupling=2.0, mainline_coupling=0.3, cross_coupling=5e-2, parasitic_coupling=5e-3),
        "cm_shifts_delta": CMShifts(self_coupling=0.1, mainline_coupling=0.05, cross_coupling=1e-3, parasitic_coupling=0),
        "samplers_size": dataset_size,
    }
    samplers_all_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                    samplers_type=sampler_type(
                                                                                        one_param=False),
                                                                                    **sampler_configs)
    samplers_all_params_shuffle_cms_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_all_params.cms.shuffle(ratio=1, dim=1),
        pss=samplers_all_params.pss,
        qs=samplers_all_params.qs,
        f0=samplers_all_params.f0,
        bw=samplers_all_params.bw
    )
    # samplers_all_params_shuffle_pss_cols = CMTheoreticalDatasetGeneratorSamplers(cms=samplers_all_params.cms,
    #                                                                                  pss=samplers_all_params.pss.shuffle(
    #                                                                                      ratio=1, dim=1),
    #                                                                              qs=samplers_all_params.qs,
    #                                                                              f0=samplers_all_params.f0)
    # samplers_all_params_flip_sign_cms_cols = CMTheoreticalDatasetGeneratorSamplers(
    #     cms=samplers_all_params.cms.flip_signs(ratio=1, dim=0),
    #     pss=samplers_all_params.pss,
    #     qs=samplers_all_params.qs
    # )
    # samplers_all_params_shuffle_all_cols = CMTheoreticalDatasetGeneratorSamplers(
    #     cms=samplers_all_params.cms.shuffle(ratio=1, dim=1),
    #     pss=samplers_all_params.pss.shuffle(ratio=1, dim=1),
    #     qs=samplers_all_params.qs
    # )

    sampler_configs["samplers_size"] = int(dataset_size / 100)
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
        total_samplers.qs._type = sampler_type
        total_samplers.f0._type = sampler_type
        total_samplers.bw._type = sampler_type
    else:
        # total_samplers = CMTheoreticalDatasetGeneratorSamplers.concat(
        #     (samplers_all_params, samplers_all_params_shuffle_pss_cols)
        # )
        total_samplers = samplers_all_params
        total_samplers.cms._type = sampler_type
        total_samplers.pss._type = sampler_type
        # total_samplers.qs._type = sampler_type
    if torch.isnan(total_samplers.cms.space).any() or torch.isinf(total_samplers.cms.space).any():
        raise ValueError("⚠️ (cms) Input to model contains NaN or Inf")
    if torch.isnan(total_samplers.pss.space).any() or torch.isinf(total_samplers.pss.space).any():
        raise ValueError("⚠️ (pss) Input to model contains NaN or Inf")
    # if torch.isnan(total_samplers.qs.space).any() or torch.isinf(total_samplers.qs.space).any():
    #     raise ValueError("⚠️ (qs) Input to model contains NaN or Inf")
    return total_samplers


def get_model(name: str="resnet_with_correction", **kwargs):
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
        main = get_model('resnet', **kwargs)
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
    elif name=="resnet_with_wide_correction":
        residual_correction_model = kwargs["main_model"]
        wide_correction_model = models.ModelWithCorrectionAndSparameters(
            main_model=residual_correction_model,
            correction_model=CorrectionNet(s_shape=(8, 301), m_dim=kwargs["out_channels"])
        )
        return wide_correction_model
    elif name=="simple_opt":
        simple_opt_model = models.Simple_Opt_3(**kwargs)
        return simple_opt_model
    elif name=="birnn":
        birnn = models.BiRNN(**kwargs)
        return birnn
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_model(model: nn.Module, work_model, dm: TouchstoneLDataModule, trainer: L.Trainer,
                loss_fn,
                optimizer_cfg:dict={"name": "AdamW", "lr": 0.0005370623202982373, "weight_decay": 1e-5},
                scheduler_cfg:dict={"name": "StepLR", "step_size": 21, "gamma": 0.01}):
    lit_model = MWFilterBaseLMWithMetrics(
        work_model=work_model,
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
    def __init__(self, ds_path, ds_size, sampler_type=SamplerTypes.SAMPLER_SOBOL):
        L.seed_everything(0)
        print("Создаем фильтр")
        self.orig_filter = create_origin_filter(configs.ENV_ORIGIN_DATA_PATH, resample_scale=301)
        print("Создаем сэмплеры")
        self.ds_size = ds_size
        self.ds_path = ds_path
        self.samplers = create_sampler(self.orig_filter, sampler_type, dataset_size=ds_size)
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
        self.model = get_model(model_name, **model_cfg)
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
            test_ratio=0.01,  # Доля тестового набора
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
        _, _, self.meta = test_tds[0]  # Используем первый файл набора данных


    def train(self, optimizer_cfg: dict, scheduler_cfg: dict, loss_fn):
        tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=f"{self.model_name}")
        csv_logger = CSVLogger(save_dir="lightning_logs", name=f"{self.model_name}_csv")
        self.trainer.loggers = [tb_logger, csv_logger]
        lit_model = train_model(model=self.model, work_model=self, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, dm=self.dm,
                    trainer=self.trainer, loss_fn=loss_fn)
        print(f"Лучшая модель сохранена в: {self.trainer.checkpoint_callback.best_model_path}")
        return lit_model

    def inference(self, path_to_ckpt: str):
        inference_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
            work_model=self,
            checkpoint_path=path_to_ckpt,
            model=self.model
        )
        return inference_model

    def predict(self, inf_model: nn.Module, dm: TouchstoneLDataModule, idx: int, with_scalers: bool=True):
        if idx == -1:  # значит предсказываем всем датасете
            predictions = [self.predict_for(inf_model, dm, i, with_scalers) for i in range(len(dm.get_dataset(split="test", meta=True)))]
        else:
            predictions = self.predict_for(inf_model, dm, idx, with_scalers)
        return predictions

    def predict_for(self, inf_model: nn.Module, dm: TouchstoneLDataModule, idx: int, with_scalers=True) -> tuple[MWFilter, MWFilter]:
        # Возьмем для примера первый touchstone-файл из тестового набора данных
        test_tds = dm.get_dataset(split="test", meta=True)
        # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
        y_t, x_t, meta = test_tds[idx]  # Используем первый файл набора данных]

        # Декодируем данные
        orig_prms = dm.codec.decode_x(x_t)  # Создаем словарь параметров
        net = dm.codec.decode_s(y_t, meta)  # Создаем объект skrf.Network

        # Предсказанные S-параметры
        pred_prms = inf_model.predict_x(net)
        if not with_scalers:
            pred_prms_vals = dm.scaler_out(torch.tensor(list(pred_prms.values())))
            orig_prms_vals = dm.scaler_out(torch.tensor(list(orig_prms.values())))
            pred_prms = dict(zip(pred_prms.keys(), list(torch.squeeze(pred_prms_vals, dim=0).numpy())))
            orig_prms = dict(zip(orig_prms.keys(), list(torch.squeeze(orig_prms_vals, dim=0).numpy())))

        print(f"Исходные параметры: {orig_prms}")
        print(f"Предсказанные параметры: {pred_prms}")

        orig_fil = MWFilter.from_touchstone_dataset_item(({**meta['params'], **orig_prms}, net))
        pred_fil = self.create_filter_from_prediction(orig_fil, self.orig_filter, pred_prms, self.codec)
        return orig_fil, pred_fil

    @staticmethod
    def create_filter_from_prediction(orig_fil: MWFilter, work_model_orig_fil: MWFilter, pred_prms: dict, codec: TouchstoneCodec) -> MWFilter:
        # Q = meta['params']['Q']
        if pred_prms.get('Q') is None:
            Q = work_model_orig_fil.Q
        else:
            Q = pred_prms['Q']
        if pred_prms.get('f0') is None:
            f0 = work_model_orig_fil.f0
        else:
            f0 = pred_prms['f0']
        if pred_prms.get('bw') is None:
            bw = work_model_orig_fil.bw
        else:
            bw = pred_prms['bw']

        fbw = bw/f0

        pred_matrix = MWFilter.matrix_from_touchstone_data_parameters(pred_prms, matrix_order=work_model_orig_fil.coupling_matrix.matrix_order)
        s_pred = MWFilter.response_from_coupling_matrix(f0=f0, FBW=fbw, frange=orig_fil.f / 1e6,
                                                        Q=Q, M=pred_matrix)
        pred_fil = MWFilter(order=work_model_orig_fil.coupling_matrix.matrix_order-2, bw=bw, f0=f0,
                            Q=Q,
                            matrix=pred_matrix, frequency=orig_fil.f, s=s_pred, z0=orig_fil.z0)
        return pred_fil

    def info(self):
        if self.codec is None:
            raise ValueError("Перед запросом информации необходимо вызвать метод setup")
        info = (f"Текущая модель: {self.model_name}\n"
               f"Частотный диапазон: {self.orig_filter.f[0]}-{self.orig_filter.f[-1]}\n"
               f"Каналы X: {self.codec.x_keys}. Количество: {len(self.codec.x_keys)}\n"
               f"Каналы Y: {self.codec.y_channels}. Количесво: {len(self.codec.y_channels)}\n")
        return info

