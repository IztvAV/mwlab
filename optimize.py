import optuna
import os

import common
import models
from configs import FILTER_NAME, BATCH_SIZE
from common import create_origin_filter
from losses import CustomLosses
from mwlab.io.backends import RAMBackend
from filters.mwfilter_lightning import MWFilterBaseLModule, MWFilterBaseLMWithMetrics
from filters.datasets import CMTheoreticalDatasetGenerator, CMTheoreticalDatasetGeneratorSamplers
from filters.datasets.theoretical_dataset_generator import PSShift, CMShifts
from filters.filter import MWFilter
from filters.codecs import MWFilterTouchstoneCodec
from filters.utils import SamplerTypes, Sampler
from mwlab import TouchstoneData, TouchstoneDataset, TouchstoneLDataModule
from mwlab.nn import MinMaxScaler
from torch import nn
import lightning as L
import pickle


DATASET_SIZE = 100_000
ENV_ORIGIN_DATA_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "origins_data")
ENV_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "optimize_data")
ENV_STUDY_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "study_results")
TRIAL_NUM = 100

ds_gen = None
codec = None


# Настраиваем, чтобы после каждого trial печатать лучшие параметры
def print_best_callback(study, trial):
    print(f"\n--- Trial #{trial.number} завершился ---")
    print(f"Best value so far: {study.best_value}")
    print(f"Best params so far: {study.best_params}\n")
    # Сохраняем на случай вылета программы
    save_study_pickle(study, path=os.path.join(ENV_STUDY_PATH,
                                               f"study_dataset-dataset={DATASET_SIZE}-trials={TRIAL_NUM}.pkl"))


def save_study_pickle(study, path):
    with open(path, 'wb') as f:
        pickle.dump(study, f)

def load_study_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def base_objective(model: nn.Module, optimizer_cfg: dict, scheduler_cfg: dict, metric: str="val_mse"):
    # 3. Создаем DataLoader
    dm = TouchstoneLDataModule(
        source=ds_gen.backend,  # Путь к датасету
        codec=codec,  # Кодек для преобразования TouchstoneData → (x, y)
        batch_size=BATCH_SIZE,  # Размер батча
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

    dm.setup("fit")

    lit_model = MWFilterBaseLMWithMetrics(
        model=model,  # Наша нейросетевая модель
        swap_xy=True,
        scaler_in=dm.scaler_in,  # Скейлер для входных данных
        scaler_out=dm.scaler_out,  # Скейлер для выходных данных
        codec=codec,  # Кодек для преобразования данных
        optimizer_cfg=optimizer_cfg,  # Конфигурация оптимизатора
        scheduler_cfg=scheduler_cfg,
        loss_fn=CustomLosses("mse_with_l1")
    )

    stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.00001)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", dirpath="optimized_models/" + FILTER_NAME,
                                                     filename="best-{epoch}-{val_loss:.5f}-{train_loss:.5f}",
                                                     mode="min",
                                                     save_top_k=1,  # Сохраняем только одну лучшую
                                                     save_weights_only=False,
                                                     # Сохранять всю модель (включая структуру)
                                                     verbose=False  # Отключаем логирование сохранения)
                                                     )

    # Обучение модели с помощью PyTorch Lightning
    trainer = L.Trainer(
        deterministic=True,
        max_epochs=100,  # Максимальное количество эпох обучения
        accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
        log_every_n_steps=100,  # Частота логирования в процессе обучения
        callbacks=[
            stoping,
            checkpoint
        ]
    )

    # Запуск процесса обучения
    trainer.fit(lit_model, dm)
    score = trainer.callback_metrics[metric].item()
    return score


def optimize_lr(metric="val_r2", direction="maximize"):
    # 2. Определяем целевую функцию для Optuna
    def objective(trial):
        # 1. Определяем пространство поиска параметров
        params = {
            'lr': trial.suggest_float('lr', 0.00001, 0.001, step=0.00001),
            'gamma': trial.suggest_float('gamma', 0.05, 1.0, step=0.05),
            'step_size': trial.suggest_int('step_size', 1, 30),
        }

        orig_filter = create_origin_filter(ENV_ORIGIN_DATA_PATH)
        # 2. Создаем модель
        main = models.ResNet1DFlexible(
            in_channels=len(codec.y_channels),
            out_channels=len(codec.x_keys),
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
            input_dim=len(codec.x_keys),
            output_dim=len(codec.x_keys),
            hidden_dims=[32, 16, 1024],
            activation_fun='soft_sign'
        )

        model = models.ModelWithCorrection(
            main_model=main,
            correction_model=mlp,
        )
        # model = models.main = models.ResNet2DFlexible(
        #     freq_vector=orig_filter.f_norm,
        #     in_channels=len(codec.y_channels),
        #     out_channels=len(codec.x_keys),
        #     use_se=False,
        #     se_reduction=1
        # )
        optimizer_cfg = {"name": "Adam", "lr": params['lr']}
        scheduler_cfg = {"name": "StepLR", "step_size": params['step_size'], "gamma": params['gamma']}
        score = base_objective(model, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, metric=metric)
        return score

    # 3. Создаем study и запускаем оптимизацию
    study = load_study_pickle(os.path.join(ENV_STUDY_PATH,
                                               f"study_dataset-dataset={DATASET_SIZE}-trials={TRIAL_NUM}.pkl"))
    # study = optuna.create_study(direction=direction)  # Мы хотим максимизировать accuracy
    study.enqueue_trial(
        {
         'lr': 0.0005995097360712593,
         'step_size': 20,
         'gamma': 0.05
         }
    )
    study.optimize(objective, n_trials=TRIAL_NUM, callbacks=[print_best_callback])  # Количество итераций оптимизации
    return study


def optimize_efficient_net(metric="val_r2", direction="maximize"):
    # 2. Определяем целевую функцию для Optuna
    def objective(trial):
        # 1. Определяем пространство поиска параметров
        params = {
            'se_ratio': trial.suggest_float('se_ratio', 0.05, 3.0, step=0.1),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.05),
            'depth_coeff': trial.suggest_float('depth_coeff', 0.1, 3.0, step=0.1),
            'width_coeff': trial.suggest_float('width_coeff', 0.1, 3.0, step=0.1),
            # 'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'lr': trial.suggest_float('lr', 1e-6, 1e-1, log=True),
            'gamma': trial.suggest_float('gamma', 0.05, 1.0, step=0.05),
            'step_size': trial.suggest_int('step_size', 1, 20),
        }

        # 2. Создаем модель
        model = models.EfficientNet1D(
            in_channels=len(codec.y_channels),  # По вашему исходному коду,
            num_classes=len(codec.x_keys),  # По вашему исходному коду
            width_coeff=params['width_coeff'],
            depth_coeff=params['depth_coeff'],
            dropout_rate=params['dropout_rate'],
            se_ratio=params['se_ratio'],
        )
        optimizer_cfg = {"name": "Adam", "lr": params['lr']}
        scheduler_cfg = {"name": "StepLR", "step_size": params['step_size'], "gamma": params['gamma']}
        score = base_objective(model, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, metric=metric)
        return score

    # 3. Создаем study и запускаем оптимизацию
    study = optuna.create_study(direction=direction)  # Мы хотим максимизировать accuracy
    study.enqueue_trial(
        {'width_coeff': 2.4,
         'depth_coeff': 0.4,
         'dropout_rate': 0.05,
         'se_ratio': 2.15,
         'lr': 0.0031623006844162787,
         'step_size': 9,
         'gamma': 0.35
         }
    )
    study.optimize(objective, n_trials=TRIAL_NUM, callbacks=[print_best_callback])  # Количество итераций оптимизации
    return study


def optimize_resnet(metric="val_r2", direction="maximize"):
    # 2. Определяем целевую функцию для Optuna
    def objective(trial):
        # 1. Определяем пространство поиска параметров
        params = {
            'first_conv_channels': trial.suggest_categorical('first_conv_channels', [16, 32, 64, 128, 256, 512, 1024]),
            'first_conv_kernel': 8,  # Нечетные размеры ядра
            'first_maxpool_kernel': 2,
            'layer_channels': [
                trial.suggest_categorical('layer1_channels', [16, 32, 64, 128, 256, 512, 1024]),
                trial.suggest_categorical('layer2_channels', [16, 32, 64, 128, 256, 512, 1024]),
                trial.suggest_categorical('layer3_channels', [16, 32, 64, 128, 256, 512, 1024]),
                trial.suggest_categorical('layer4_channels', [16, 32, 64, 128, 256, 512, 1024]),
            ],
            'num_blocks': [
                trial.suggest_int('layer1_blocks', 1, 6),
                trial.suggest_int('layer2_blocks', 1, 6),
                trial.suggest_int('layer3_blocks', 1, 6),
                trial.suggest_int('layer4_blocks', 1, 6),
            ],
            # 'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, step=0.01),
            'step_size': trial.suggest_int('step_size', 5, 25),
            'activation_in': trial.suggest_categorical('activation_in', models.get_available_activations()),
            'activation_block': trial.suggest_categorical('activation_block', models.get_available_activations())
        }

        # 2. Создаем модель
        model = models.ResNet1DFlexible(
            in_channels=len(codec.y_channels),  # По вашему исходному коду
            out_channels=len(codec.x_keys),  # По вашему исходному коду
            first_conv_channels=params['first_conv_channels'],
            first_conv_kernel=params['first_conv_kernel'],
            layer_channels=params['layer_channels'],
            num_blocks=params['num_blocks'],
            activation_in=params['activation_in'],
            activation_block=params['activation_block']
        )
        optimizer_cfg = {"name": "Adam", "lr": params['lr']}
        scheduler_cfg = {"name": "StepLR", "step_size": params['step_size'], "gamma": params['gamma']}
        score = base_objective(model, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, metric=metric)
        return score

    # 3. Создаем study и запускаем оптимизацию
    study = optuna.create_study(direction=direction)  # Мы хотим максимизировать accuracy
    study.enqueue_trial(
        {'first_conv_channels': 64,
         'first_conv_kernel': 8,
         'layer1_channels': 64,
         'layer2_channels': 64,
         'layer3_channels': 128,
         'layer4_channels': 256,
         'layer1_blocks': 1,
         'layer2_blocks': 4,
         'layer3_blocks': 3,
         'layer4_blocks': 5,
         'lr': 0.0005587648891507119,
         'gamma': 0.1,
         'step_size': 20,
         'activation_in': 'sigmoid',
         'activation_block': 'swish'}
    )
    study.optimize(objective, n_trials=TRIAL_NUM, callbacks=[print_best_callback])  # Количество итераций оптимизации
    return study


def optimize_resnet_with_mlp_correction(metric: str="val_mse", direction: str="minimize"):
    def objective(trial):
        # 1. Определяем пространство поиска параметров
        params = {
            'hidden_dims': [
                trial.suggest_categorical('hidden1_features', [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
                trial.suggest_categorical('hidden2_features', [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
                trial.suggest_categorical('hidden3_features', [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
            ],
            'lr': trial.suggest_float('lr', 1e-6, 1e-1, log=True),
            'gamma': trial.suggest_float('gamma', 0.05, 1.0, step=0.05),
            'step_size': trial.suggest_int('step_size', 1, 20, step=1),
            'activation_fun': trial.suggest_categorical('activation_fun', models.get_available_activations()),
        }

        # 2. Создаем модель
        main = models.ResNet1DFlexible(
            in_channels=len(codec.y_channels),
            out_channels=len(codec.x_keys),
            num_blocks=[1, 4, 3, 5],
            layer_channels=[64, 64, 128, 256],
            first_conv_kernel=8,
            first_maxpool_kernel=2,
            first_conv_channels=64,
            activation_in='sigmoid',
            activation_block='swish'
        )

        correction = models.CorrectionMLP(
            input_dim=len(codec.x_keys),
            output_dim=len(codec.x_keys),
            hidden_dims=params['hidden_dims'],
            activation_fun=params['activation_fun']
        )

        model = models.ModelWithCorrection(
            main_model=main,
            correction_model=correction
        )
        optimizer_cfg = {"name": "AdamW", "lr": params['lr'], "weight_decay": 1e-5}
        scheduler_cfg = {"name": "StepLR", "step_size": params['step_size'], "gamma": params['gamma']}
        score = base_objective(model, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, metric=metric)
        return score

    # 3. Создаем study и запускаем оптимизацию
    study = optuna.create_study(direction=direction)  # Мы хотим максимизировать accuracy
    study.enqueue_trial(
        {'hidden1_features': 128,
         'hidden2_features': 256,
         'hidden3_features': 512,
         'lr': 0.0005587648891507119,
         'gamma': 0.1,
         'step_size': 15,
         'activation_fun': 'swish'
         }
    )
    study.optimize(objective, n_trials=TRIAL_NUM, callbacks=[print_best_callback])  # Количество итераций оптимизации
    return study


def main():
    # 1. Загрузка данных
    print("Создаем фильтр")
    orig_filter = create_origin_filter(ENV_ORIGIN_DATA_PATH)
    print("Создаем сэмплеры")
    samplers = common.create_sampler(orig_filter, SamplerTypes.SAMPLER_SOBOL, dataset_size=DATASET_SIZE)
    global ds_gen
    ds_gen = CMTheoreticalDatasetGenerator(
        path_to_save_dataset=os.path.join(ENV_DATASET_PATH, samplers.cms.type.name, f"{len(samplers.cms)}"),
        backend_type='ram',
        orig_filter=orig_filter,
        filename="Dataset",
    )
    ds_gen.generate(samplers)


    global codec
    codec = MWFilterTouchstoneCodec.from_dataset(ds=TouchstoneDataset(source=ds_gen.backend, in_memory=True),
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in orig_filter.coupling_matrix.links])
    codec.exclude_keys(["f0", "bw", "N", "Q"])
    print(codec)

    # Исключаем из анализа ненужные x-параметры
    print("Каналы:", codec.y_channels)
    print("Количество каналов:", len(codec.y_channels))

    study = optimize_resnet_with_mlp_correction(metric="val_mae", direction="minimize")

    # 4. Выводим результаты
    print(f"Лучшие параметры: {study.best_params}")
    print(f"Лучшая метрика: {study.best_value:.4f}")

    if not os.path.exists(ENV_STUDY_PATH):
        os.makedirs(ENV_STUDY_PATH)
    save_study_pickle(study, path=os.path.join(ENV_STUDY_PATH,
                                               f"study_dataset-dataset={DATASET_SIZE}-trials={TRIAL_NUM}.pkl"))
    # 5. Визуализация (опционально)
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()


if __name__ == "__main__":
    L.seed_everything(0)
    main()
