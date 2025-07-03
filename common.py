import os

import torch
from torch import nn

from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLMWithMetrics
from mwlab import TouchstoneDataset, TouchstoneDatasetAnalyzer, TouchstoneLDataModule
from mwlab.nn import MinMaxScaler
from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample

from filters import CMTheoreticalDatasetGeneratorSamplers, SamplerTypes, MWFilter, CouplingMatrix
from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift, CMTheoreticalDatasetGenerator
import models
import configs
import lightning as L


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
        f_start = f0 - 2 * bw
    if f_stop is None:
        f_stop = f0 + 2 * bw
    if f_unit is None:
        f_unit = "MHz"
    y_transform = TComposite([
        S_Crop(f_start=f_start, f_stop=f_stop, unit=f_unit),
        S_Resample(resample_scale)
    ])
    tds_transformed = TouchstoneDataset(source=path_orig_filter, s_tf=y_transform)
    origin_filter = MWFilter.from_touchstone_dataset_item(tds_transformed[0])
    return origin_filter


def create_sampler(orig_filter: MWFilter, sampler_type: SamplerTypes, with_one_param: bool=False, dataset_size=configs.BASE_DATASET_SIZE):
    sampler_configs = {
        "pss_origin": PSShift(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017),
        "pss_shifts_delta": PSShift(phi11=0.02, phi21=0.02, theta11=0.005, theta21=0.005),
        "cm_shifts_delta": CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=5e-3, parasitic_coupling=5e-3),
        "samplers_size": dataset_size,
    }
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
    samplers_all_params_shuffle_all_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_all_params.cms.shuffle(ratio=1, dim=1),
        pss=samplers_all_params.pss.shuffle(ratio=1, dim=1)
    )

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
    else:
        total_samplers = CMTheoreticalDatasetGeneratorSamplers.concat(
            (samplers_all_params, samplers_all_params_shuffle_cms_cols)
        )
        total_samplers.cms._type = sampler_type
        total_samplers.pss._type = sampler_type
    if torch.isnan(total_samplers.cms.space).any() or torch.isinf(total_samplers.cms.space).any():
        raise ValueError("⚠️ (cms) Input to model contains NaN or Inf")
    if torch.isnan(total_samplers.pss.space).any() or torch.isinf(total_samplers.pss.space).any():
        raise ValueError("⚠️ (pss) Input to model contains NaN or Inf")
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
        main = models.ResNet1DFlexible(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            num_blocks=[1, 4, 3, 5],
            layer_channels=[64, 64, 128, 256],
            first_conv_kernel=8,
            first_conv_channels=64,
            first_maxpool_kernel=3,
            activation_in='sigmoid',
            activation_block='swish',
            use_se=False,
            se_reduction=1
        )
        mlp = models.CorrectionMLP(
            input_dim=kwargs["out_channels"],
            output_dim=kwargs["out_channels"],
            hidden_dims=[128, 256, 512],
            activation_fun='swish'
        )
        model = models.ModelWithCorrection(
            main_model=main,
            correction_model=mlp,
        )
        return model
    else:
        raise ValueError(f"Unknown model name: {name}")


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


class WorkModel:
    CHECKPOINT_DIRPATH = "saved_models/" + configs.FILTER_NAME
    BEST_MODEL_FILENAME_SUFFIX = "-batch_size={batch_size}-base_dataset_size={base_dataset_size}-sampler={sampler_type}"
    def __init__(self):
        L.seed_everything(0)
        print("Создаем фильтр")
        self.orig_filter = create_origin_filter(configs.ENV_ORIGIN_DATA_PATH, resample_scale=301)
        print("Создаем сэмплеры")
        self.samplers = create_sampler(self.orig_filter, SamplerTypes.SAMPLER_SOBOL)
        self.ds_gen = CMTheoreticalDatasetGenerator(
            path_to_save_dataset=os.path.join(configs.ENV_DATASET_PATH, self.samplers.cms.type.name, f"{len(self.samplers.cms)}"),
            backend_type='ram',
            orig_filter=self.orig_filter,
            filename="Dataset",
        )
        self.ds_gen.generate(self.samplers)

        self.ds = TouchstoneDataset(source=self.ds_gen.backend, in_memory=True)

        self.trainer = self._configure_trainer()
        self.codec = MWFilterTouchstoneCodec.from_dataset(ds=self.ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in self.orig_filter.coupling_matrix.links])
        self.model = None
        self.dm = None

    def _configure_trainer(self):
        stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=31, mode="min", min_delta=0.00001)
        checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss",
                                                         dirpath="saved_models/" + configs.FILTER_NAME,
                                                         filename="best-{epoch}-{train_loss:.5f}-{val_loss:.5f}-{val_r2:.5f}-{val_mse:.5f}-"
                                                                  "{val_mae:.5f}" + self.BEST_MODEL_FILENAME_SUFFIX.format(
                                                             batch_size=configs.BATCH_SIZE,
                                                             base_dataset_size=configs.BASE_DATASET_SIZE,
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
            max_epochs=150,  # Максимальное количество эпох обучения
            accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
            log_every_n_steps=100,  # Частота логирования в процессе обучения
            callbacks=[stoping, checkpoint]
        )
        return trainer

    def setup(self, model_name: str, model_cfg: dict, dm_codec: MWFilterTouchstoneCodec):
        self.model = get_model(model_name, **model_cfg)
        print(dm_codec)
        print("Каналы Y:", dm_codec.y_channels)
        print("Каналы X:", dm_codec.x_keys)
        print("Количество каналов:", len(dm_codec.y_channels))
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

    def train(self, optimizer_cfg: dict, scheduler_cfg: dict, loss_fn):
        lit_model = train_model(model=self.model, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, dm=self.dm,
                    trainer=self.trainer, loss_fn=loss_fn)
        return lit_model