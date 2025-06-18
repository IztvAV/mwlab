from torch import nn

from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_lightning import MWFilterBaseLMWithMetrics
from mwlab import TouchstoneDataset, TouchstoneDatasetAnalyzer, TouchstoneLDataModule
from mwlab.transforms import TComposite
from mwlab.transforms.s_transforms import S_Crop, S_Resample

from filters import CMTheoreticalDatasetGeneratorSamplers, SamplerTypes, MWFilter, CouplingMatrix
from filters.datasets.theoretical_dataset_generator import CMShifts, PSShift
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
        f_start = f0 - 1.2 * bw
    if f_stop is None:
        f_stop = f0 + 1.2 * bw
    if f_unit is None:
        f_unit = "MHz"
    y_transform = TComposite([
        S_Crop(f_start=f_start, f_stop=f_stop, unit=f_unit),
        S_Resample(resample_scale)
    ])
    tds_transformed = TouchstoneDataset(source=path_orig_filter, s_tf=y_transform)
    origin_filter = MWFilter.from_touchstone_dataset_item(tds_transformed[0])
    return origin_filter


def create_lhs_samplers(orig_filter: MWFilter):
    sampler_configs = {
        "pss_origin": PSShift(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017),
        "pss_shifts_delta": PSShift(phi11=0.02, phi21=0.02, theta11=0.005, theta21=0.005),
        "cm_shifts_delta": CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=0.005),
        "samplers_size": configs.BASE_DATASET_SIZE,
    }
    samplers_lhs_all_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                    samplers_type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE(one_param=False),
                                                                                    **sampler_configs)
    samplers_lhs_all_params_shuffle_cms_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_lhs_all_params.cms.shuffle(ratio=1, dim=1),
        pss=samplers_lhs_all_params.pss)
    samplers_lhs_all_params_shuffle_pss_cols = CMTheoreticalDatasetGeneratorSamplers(cms=samplers_lhs_all_params.cms,
                                                                                pss=samplers_lhs_all_params.pss.shuffle(
                                                                                ratio=1, dim=1))
    samplers_lhs_all_params_shuffle_all_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_lhs_all_params.cms.shuffle(ratio=1, dim=1),
        pss=samplers_lhs_all_params.pss.shuffle(ratio=1, dim=1)
    )

    sampler_configs["samplers_size"] = int(configs.BASE_DATASET_SIZE/100)
    samplers_lhs_with_one_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                         samplers_type=SamplerTypes.SAMPLER_LATIN_HYPERCUBE(one_param=True),
                                                                                         **sampler_configs)
    total_samplers = CMTheoreticalDatasetGeneratorSamplers.concat(
            (samplers_lhs_all_params_shuffle_cms_cols, samplers_lhs_all_params_shuffle_pss_cols, samplers_lhs_all_params_shuffle_all_cols,
             samplers_lhs_all_params)
    )
    return total_samplers


def create_std_samplers(orig_filter: MWFilter):
    sampler_configs = {
        "pss_origin": PSShift(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017),
        "pss_shifts_delta": PSShift(phi11=0.02, phi21=0.02, theta11=0.005, theta21=0.005),
        "cm_shifts_delta": CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=0.005),
        "samplers_size": configs.BASE_DATASET_SIZE,
    }
    samplers_std_all_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                    samplers_type=SamplerTypes.SAMPLER_STD(one_param=False),
                                                                                    **sampler_configs)
    samplers_lhs_all_params_shuffle_cms_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_std_all_params.cms.shuffle(ratio=1, dim=1),
        pss=samplers_std_all_params.pss)
    samplers_lhs_all_params_shuffle_pss_cols = CMTheoreticalDatasetGeneratorSamplers(cms=samplers_std_all_params.cms,
                                                                                     pss=samplers_std_all_params.pss.shuffle(
                                                                                         ratio=1, dim=1))
    samplers_lhs_all_params_shuffle_all_cols = CMTheoreticalDatasetGeneratorSamplers(
        cms=samplers_std_all_params.cms.shuffle(ratio=1, dim=1),
        pss=samplers_std_all_params.pss.shuffle(ratio=1, dim=1)
    )

    sampler_configs["samplers_size"] = int(configs.BASE_DATASET_SIZE / 100)
    samplers_lhs_with_one_params = CMTheoreticalDatasetGeneratorSamplers.create_samplers(orig_filter,
                                                                                         samplers_type=SamplerTypes.SAMPLER_STD(
                                                                                             one_param=True),
                                                                                         **sampler_configs)
    total_samplers = CMTheoreticalDatasetGeneratorSamplers.concat(
        (samplers_lhs_all_params_shuffle_cms_cols, samplers_lhs_all_params_shuffle_pss_cols,
         samplers_lhs_all_params_shuffle_all_cols,
         samplers_std_all_params)
    )
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
    elif name == "resnet_with_correction":
        main = models.ResNet1DFlexible(
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

        mlp = models.CorrectionMLP(
            input_dim=kwargs["out_channels"],
            output_dim=kwargs["out_channels"],
            hidden_dims=[8, 64, 4096],
            activation_fun='soft_sign'
        )

        model = models.ModelWithCorrection(
            main_model=main,
            correction_model=mlp,
        )

        return model


def train_model(model: nn.Module, dm: TouchstoneLDataModule,
                stop_cb=None,
                checkpoint_cb=None):
    dm.setup("fit")

    # Декодирование
    _, __, meta = dm.get_dataset(split="train", meta=True)[0]

    # Печатаем размеры полученных наборов
    print(f"Размер тренировочного набора: {len(dm.train_ds)}")
    print(f"Размер валидационного набора: {len(dm.val_ds)}")
    print(f"Размер тестового набора: {len(dm.test_ds)}")

    lit_model = MWFilterBaseLMWithMetrics(
        model=model,  # Наша нейросетевая модель
        swap_xy=True,
        scaler_in=dm.scaler_in,  # Скейлер для входных данных
        scaler_out=dm.scaler_out,  # Скейлер для выходных данных
        codec=dm.codec,  # Кодек для преобразования данных
        optimizer_cfg={"name": "Adam", "lr": 0.0007526812333573349},
        scheduler_cfg={"name": "StepLR", "step_size": 14, "gamma": 0.15},
        # loss_fn=CustomLosses("error"),
        loss_fn=nn.MSELoss()
    )

    callbacks = []
    if checkpoint_cb is not None:
        print("Set checkpoint callback")
        callbacks.append(checkpoint_cb)
    if stop_cb is not None:
        print("Set stop callback")
        callbacks.append(stop_cb)
    if not callbacks:
        print("Callbacks is empty!!!")

    # Обучение модели с помощью PyTorch Lightning
    trainer = L.Trainer(
        deterministic=True,
        max_epochs=150,  # Максимальное количество эпох обучения
        accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
        log_every_n_steps=100,  # Частота логирования в процессе обучения
        callbacks=callbacks
    )

    trainer.fit(lit_model)
    return lit_model
