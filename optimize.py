import optuna
import os
import models
from main import FILTER_NAME, BATCH_SIZE
from mwlab.io.backends import RAMBackend
from filters.mwfilter_lightning import MWFilterBaseLModule, MWFilterBaseLMWithMetrics
from filters.datasets import CMTheoreticalDatasetGenerator
from filters.datasets.theoretical_dataset_generator import PSShift, CMShifts
from filters.filter import MWFilter
from filters.codecs import MWFilterTouchstoneCodec
from mwlab import TouchstoneData, TouchstoneDataset, TouchstoneLDataModule
from mwlab.nn import MinMaxScaler
from torch import nn
import lightning as L
import pickle


DATASET_SIZE = 500_000
ENV_ORIGIN_DATA_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "origins_data")
ENV_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "optimize_data")
ENV_STUDY_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "study_results")
TRIAL_NUM = 100

backend = None
codec = None


def save_study_pickle(study, path):
    with open(path, 'wb') as f:
        pickle.dump(study, f)

def load_study_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# 2. Определяем целевую функцию для Optuna
def objective(trial):
    # 1. Определяем пространство поиска параметров
    params = {
        'first_conv_channels': trial.suggest_categorical('first_conv_channels', [16, 32, 64, 128]),
        'first_conv_kernel': trial.suggest_int('first_conv_kernel', 5, 11, step=1),  # Нечетные размеры ядра
        'layer_channels': [
            trial.suggest_categorical('layer1_channels', [16, 32, 64, 128]),
            trial.suggest_categorical('layer2_channels', [32, 64, 128, 256]),
            trial.suggest_categorical('layer3_channels', [64, 128, 256, 512]),
            trial.suggest_categorical('layer4_channels', [128, 256, 512, 1024]),
        ],
        'num_blocks': [
            trial.suggest_int('layer1_blocks', 1, 5),
            trial.suggest_int('layer2_blocks', 1, 5),
            trial.suggest_int('layer3_blocks', 1, 5),
            trial.suggest_int('layer4_blocks', 1, 5),
        ],
        # 'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0, step=0.1),
        'step_size': trial.suggest_int('step_size', 1, 30)
    }

    # 2. Создаем модель
    model = models.ResNet1DFlexible(
        in_channels=8,  # По вашему исходному коду
        out_channels=30,  # По вашему исходному коду
        first_conv_channels=params['first_conv_channels'],
        first_conv_kernel=params['first_conv_kernel'],
        layer_channels=params['layer_channels'],
        num_blocks=params['num_blocks'],
    )

    # 3. Создаем DataLoader
    dm = TouchstoneLDataModule(
        source=backend,  # Путь к датасету
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
        optimizer_cfg={"name": "Adam", "lr": params['lr']},  # Конфигурация оптимизатора
        scheduler_cfg={"name": "StepLR", "step_size": params['step_size'], "gamma": params['gamma']},
        loss_fn=nn.MSELoss()
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
        max_epochs=500,  # Максимальное количество эпох обучения
        accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
        log_every_n_steps=100,  # Частота логирования в процессе обучения
        callbacks=[
            stoping,
            checkpoint
        ]
    )

    # Запуск процесса обучения
    trainer.fit(lit_model, dm)

    val_loss = checkpoint.best_model_score
    score = val_loss

    return score


def main():
    # 1. Загрузка данных
    global backend
    backend = RAMBackend([])
    ds_gen = CMTheoreticalDatasetGenerator(
        path_to_origin_filter=ENV_ORIGIN_DATA_PATH,
        path_to_save_dataset=ENV_DATASET_PATH,
        pss_origin=PSShift(phi11=0.547, phi21=-1.0, theta11=0.01685, theta21=0.017),
        pss_shifts_delta=PSShift(phi11=0.02, phi21=0.02, theta11=0.005, theta21=0.005),
        cm_shifts_delta=CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=0.005),
        samplers_size=DATASET_SIZE,
        backend=backend
    )
    ds_gen.generate()

    global codec
    codec = MWFilterTouchstoneCodec.from_dataset(TouchstoneDataset(source=backend, in_memory=True))
    codec.exclude_keys(["f0", "bw", "N", "Q"])
    print(codec)

    # Исключаем из анализа ненужные x-параметры
    print("Каналы:", codec.y_channels)
    print("Количество каналов:", len(codec.y_channels))

    # 3. Создаем study и запускаем оптимизацию
    study = optuna.create_study(direction='minimize')  # Мы хотим максимизировать accuracy
    study.enqueue_trial(
        {'first_conv_channels': 128,
         'first_conv_kernel': 11,
         'layer1_channels': 128,
         'layer2_channels': 256,
         'layer3_channels': 512,
         'layer4_channels': 512,
         'layer1_blocks': 1,
         'layer2_blocks': 1,
         'layer3_blocks': 5,
         'layer4_blocks': 3,
         'lr': 0.0017552306729777972,
         'gamma': 0.1,
         'step_size': 10}
    )
    study.optimize(objective, n_trials=TRIAL_NUM)  # Количество итераций оптимизации

    # 4. Выводим результаты
    print(f"Лучшие параметры: {study.best_params}")
    print(f"Лучшая accuracy: {study.best_value:.4f}")

    if not os.path.exists(ENV_STUDY_PATH):
        os.makedirs(ENV_STUDY_PATH)
    save_study_pickle(study, path=os.path.join(ENV_STUDY_PATH, f"study_dataset-dataset={DATASET_SIZE}-trials={TRIAL_NUM}.pkl"))
    # 5. Визуализация (опционально)
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()


if __name__ == "__main__":
    main()
