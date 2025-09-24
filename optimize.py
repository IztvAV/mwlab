import optuna
import os

import common
import models
from configs import FILTER_NAME, BATCH_SIZE
from common import create_origin_filter, WorkModel
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
ENV_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "optimize_data", "ae")
ENV_STUDY_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "study_results", "ae")
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


def base_objective(work_model: WorkModel, params, metric: str="val_mse", direction="minimize", max_epoch: int=50):
    if direction == "minimize":
        mode="min"
    elif direction == "maximize":
        mode="max"
    else:
        raise ValueError("Unsupported optimize direction")
    stoping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=params['step_size']+5, mode="min", min_delta=0.00001)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor=metric, dirpath="optimized_models/ae/" + FILTER_NAME,
                                                     filename="best-{epoch}-{val_loss:.5f}-{train_loss:.5f}",
                                                     mode=mode,
                                                     save_top_k=1,  # Сохраняем только одну лучшую
                                                     save_weights_only=False,
                                                     # Сохранять всю модель (включая структуру)
                                                     verbose=False  # Отключаем логирование сохранения)
                                                     )

    trainer = L.Trainer(
        deterministic=True,
        max_epochs=max_epoch,  # Максимальное количество эпох обучения
        accelerator="auto",  # Автоматический выбор устройства (CPU/GPU)
        log_every_n_steps=100,  # Частота логирования в процессе обучения
        callbacks=[stoping, checkpoint]
    )

    work_model.trainer = trainer

    # Запуск процесса обучения
    lit_model = work_model.train(
        optimizer_cfg={"name": "AdamW", "lr": params['lr'], "weight_decay": 1e-2},
        scheduler_cfg={"name": "StepLR", "step_size": params['step_size'], "gamma": params['gamma']},
        loss_fn=CustomLosses("log_cosh")
    )
    # score = trainer.callback_metrics[metric].item()
    score = checkpoint.best_model_score
    return score


def optimize_lr(work_model: WorkModel, metric="val_r2", direction="maximize"):
    # 2. Определяем целевую функцию для Optuna
    def objective(trial):
        # 1. Определяем пространство поиска параметров
        params = {
            'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, step=0.01),
            'step_size': trial.suggest_int('step_size', 1, 35),
        }

        work_model.setup(
            model_name="imp_cae",
            model_cfg={"in_ch": len(work_model.codec.y_channels), "z_dim": work_model.orig_filter.order * 4},
            dm_codec=work_model.codec
        )

        # score = work_model.trainer.checkpoint_callback.best_model_score
        score = base_objective(work_model=work_model, params=params, max_epoch=35, direction=direction, metric=metric)
        return score

    # 3. Создаем study и запускаем оптимизацию
    study = optuna.create_study(direction=direction)  # Мы хотим максимизировать accuracy
    study.enqueue_trial(
        {
         'lr': 0.0005371,
         'step_size': 26,
         'gamma': 0.1
         }
    )
    study.optimize(objective, n_trials=TRIAL_NUM, callbacks=[print_best_callback])  # Количество итераций оптимизации
    return study


def optimize_resnet(metric="val_r2", direction="maximize"):
    # 2. Определяем целевую функцию для Optuna
    def objective(trial):
        params = {
            'first_conv_channels': trial.suggest_categorical('first_conv_channels', [16, 32, 64, 128, 256, 512, 1024]),
            'first_conv_kernel': trial.suggest_int('first_conv_kernel', 2, 10, step=1),  # Нечетные размеры ядра
            'first_maxpool_kernel': trial.suggest_int('first_maxpool_kernel', 2, 10, step=1),
            'layer_channels': [
                trial.suggest_categorical('layer1_channels', [16, 32, 64, 128, 256, 512, 1024]),
                trial.suggest_categorical('layer2_channels', [16, 32, 64, 128, 256, 512, 1024]),
                trial.suggest_categorical('layer3_channels', [16, 32, 64, 128, 256, 512, 1024]),
                # trial.suggest_categorical('layer4_channels', [16, 32, 64, 128, 256, 512, 1024]),
            ],
            'num_blocks': [
                trial.suggest_int('layer1_blocks', 1, 6),
                trial.suggest_int('layer2_blocks', 1, 6),
                trial.suggest_int('layer3_blocks', 1, 6),
                # trial.suggest_int('layer4_blocks', 1, 6),
            ],
            # 'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, step=0.01),
            'step_size': trial.suggest_int('step_size', 5, 25),
            'activation_in': trial.suggest_categorical('activation_in', models.get_available_activations()),
            'activation_block': trial.suggest_categorical('activation_block', models.get_available_activations())
        }
        # 1. Определяем пространство поиска параметров

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
        score = base_objective(model, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, metric=metric, direction=direction)
        return score

    # 3. Создаем study и запускаем оптимизацию
    study = optuna.create_study(direction=direction)  # Мы хотим максимизировать accuracy
    # study.enqueue_trial(
    #     {'first_conv_channels': 64,
    #      'first_conv_kernel': 8,
    #      'layer1_channels': 64,
    #      'layer2_channels': 64,
    #      'layer3_channels': 128,
    #      'layer4_channels': 256,
    #      'layer1_blocks': 1,
    #      'layer2_blocks': 4,
    #      'layer3_blocks': 3,
    #      'layer4_blocks': 5,
    #      'lr': 0.0005587648891507119,
    #      'gamma': 0.1,
    #      'step_size': 20,
    #      'activation_in': 'sigmoid',
    #      'activation_block': 'swish'}
    # )
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
            # 'lr': trial.suggest_float('lr', 1e-6, 1e-1, log=True),
            # 'gamma': trial.suggest_float('gamma', 0.05, 1.0, step=0.05),
            # 'step_size': trial.suggest_int('step_size', 1, 20, step=1),
            'activation_fun': trial.suggest_categorical('activation_fun', models.get_available_activations()),
        }

        # 2. Создаем модель
        # resnet = models.ResNet1DFlexible(
        #     in_channels=len(codec.y_channels),
        #     out_channels=len(codec.x_keys),
        #     num_blocks=[7, 7, 1, 3],
        #     layer_channels=[512, 64, 64, 32],
        #     first_conv_kernel=11,
        #     first_conv_channels=1024,
        #     first_maxpool_kernel=3,
        #     activation_in='leaky_relu',
        #     activation_block='leaky_relu',
        #     # use_se=False,
        #     # se_reduction=1
        # )
        resnet = common.get_model('resnet', in_channels=len(codec.y_channels), out_channels=len(codec.x_keys))

        correction = models.CorrectionMLP(
            input_dim=len(codec.x_keys),
            output_dim=len(codec.x_keys),
            hidden_dims=params['hidden_dims'],
            activation_fun=params['activation_fun']
        )

        model = models.ModelWithCorrection(
            main_model=resnet,
            correction_model=correction
        )
        optimizer_cfg = {"name": "AdamW", "lr": 0.0005371, "weight_decay": 1e-5}
        scheduler_cfg = {"name": "StepLR", "step_size": 24, "gamma": 0.01}
        score = base_objective(model, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg, metric=metric, max_epoch=scheduler_cfg['step_size']+5, direction=direction)
        return score

    # 3. Создаем study и запускаем оптимизацию
    study = optuna.create_study(direction=direction)  # Мы хотим максимизировать accuracy
    study.enqueue_trial(
        {'hidden1_features': 128,
         'hidden2_features': 256,
         'hidden3_features': 512,
         # 'lr': 0.0005587648891507119,
         # 'gamma': 0.1,
         # 'step_size': 15,
         'activation_fun': 'swish'
         }
    )
    study.optimize(objective, n_trials=TRIAL_NUM, callbacks=[print_best_callback])  # Количество итераций оптимизации
    return study


def main():
    sampler_configs = {
        "pss_origin": PSShift(phi11=0.547, phi21=-1.0, theta11=0.1685, theta21=0.17),
        "pss_shifts_delta": PSShift(phi11=0.2, phi21=0.2, theta11=0.05, theta21=0.05),
        "cm_shifts_delta": CMShifts(self_coupling=1.5, mainline_coupling=0.1, cross_coupling=5e-3,
                                    parasitic_coupling=5e-3),
        "samplers_size": DATASET_SIZE,
    }
    work_model = common.AEWorkModel(ENV_DATASET_PATH, sampler_configs, SamplerTypes.SAMPLER_SOBOL)
    # common.plot_distribution(work_model.ds, num_params=len(work_model.ds_gen.origin_filter.coupling_matrix.links))
    # plt.show()

    codec = MWFilterTouchstoneCodec.from_dataset(ds=work_model.ds,
                                                 keys_for_analysis=[f"m_{r}_{c}" for r, c in
                                                                    work_model.orig_filter.coupling_matrix.links])
    codec.y_channels = ['S1_1.db', 'S1_2.db', 'S2_1.db', 'S2_2.db']
    codec = codec
    work_model.setup(
        model_name="imp_cae",
        model_cfg={"in_ch": len(codec.y_channels), "z_dim": work_model.orig_filter.order * 4},
        dm_codec=codec
    )
    os.makedirs(ENV_STUDY_PATH, exist_ok=True)
    # study = load_study_pickle(path=os.path.join(ENV_STUDY_PATH,
    #                                            f"study_dataset-dataset={DATASET_SIZE}-trials={TRIAL_NUM}.pkl"))
    # Исключаем из анализа ненужные x-параметры
    print("Каналы:", codec.y_channels)
    print("Количество каналов:", len(codec.y_channels))

    study = optimize_lr(work_model=work_model, metric="val_r2", direction="maximize")

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
