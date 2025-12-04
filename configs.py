# import os
#
# F_START_MHZ = 11489.7
# F_STOP_MHZ = 11590.7
# # F_START_MHZ = 12695
# # F_STOP_MHZ = 12755
#
# BATCH_SIZE = 32
# BASE_DATASET_SIZE = 1_000
# # FILTER_NAME = "EAMU4T1-BPFC2"
# # FILTER_NAME = "ERV-KuIMUXT1-BPFC1"
# # FILTER_NAME = "EAMU4-KuIMUXT3-BPFC1"
# FILTER_NAME = "EAMU4-KuIMUXT2-BPFC2"
# # FILTER_NAME = "EAMU4-KuIMUXT2-BPFC4"
# # FILTER_NAME = "SCYA501-KuIMUXT5-BPFC3"
#
# ENV_DEFAULT_FILTER_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME)
# ENV_ORIGIN_DATA_PATH = os.path.join(ENV_DEFAULT_FILTER_PATH, "origins_data")
# ENV_DATASET_PATH = os.path.join(ENV_DEFAULT_FILTER_PATH, "datasets_data")
# ENV_TUNE_DATASET_PATH = os.path.join(ENV_DEFAULT_FILTER_PATH, "tune_data")
# ENV_SAVED_MODELS_PATH = os.path.join(ENV_DEFAULT_FILTER_PATH, "saved_models")


import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# =========
# Датаклассы под manifest.yml
# =========

@dataclass
class MetaConfig:
    author: str


@dataclass
class FilterConfig:
    name: str
    f_start_mhz: float
    f_stop_mhz: float


@dataclass
class PathsConfig:
    datasets_data: str
    origins_data: str
    tune_data: str
    saved_models: str


@dataclass
class DatasetConfig:
    size: int  # раньше BASE_DATASET_SIZE
    sampler_type: str


@dataclass
class ModelConfig:
    checkpoint: str  # путь внутри архива/каталога


@dataclass
class InferenceConfig:
    batch_size: int
    device: str  # "auto", "cuda", "cpu" и т.п.


@dataclass
class AppConfig:
    """
    Полная конфигурация, соответствующая manifest.yml.
    base_dir — корневая директория фильтра (там лежат подкаталоги datasets_data, origins_data и т.п.)
    """
    # version: int
    meta: MetaConfig
    filter: FilterConfig
    paths: PathsConfig
    dataset: DatasetConfig
    model: ModelConfig
    inference: InferenceConfig

    base_dir: Path  # добавляем поле, чтобы потом легко получать ENV_* пути


# =========
# Функция загрузки manifest.yml
# =========

# =========
# Формируем старые константы
# =========


# APP_CONFIG: AppConfig = init_manifest_configs(MANIFEST_PATH)

# ---- Поля, которые раньше были константами в configs.py ----


class Configs:
    def __init__(self, manifest_path: str | os.PathLike):
        # Дополнительно: путь к чекпоинту модели (если нужно где-то использовать)
        self.INFERENCE_DEVICE: str | None = None
        self.APP_CONFIG: AppConfig = self.load_app_configs(manifest_path)

    @classmethod
    def init_as_default(cls, default_path: str | os.PathLike):
        manifest_path = cls._get_manifest_path_from_default(default_path)
        return cls(manifest_path)

    @staticmethod
    def _get_manifest_path_from_default(manifest_path: os.PathLike | str) -> str | os.PathLike:
        manifest_path = Path(manifest_path).resolve()

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        manifest_path = str(manifest_path.parent / data["paths"]["manifest"])
        return manifest_path


    @staticmethod
    def load_app_configs(manifest_path: os.PathLike | str) -> AppConfig:
        manifest_path = Path(manifest_path).resolve()

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # version = data.get("version", 1)
        # if version != 1:
        #     raise ValueError(f"Unsupported manifest version: {version}")

        # Корневая директория фильтра — там же лежит manifest.yml
        base_dir = manifest_path.parent

        meta_cfg = MetaConfig(**data["meta"])
        filter_cfg = FilterConfig(**data["filter"])
        paths_cfg = PathsConfig(**data["paths"])
        dataset_cfg = DatasetConfig(**data["dataset"])
        model_cfg = ModelConfig(**data["model"])
        inference_cfg = InferenceConfig(**data["inference"])

        return AppConfig(
            # version=version,
            meta=meta_cfg,
            filter=filter_cfg,
            paths=paths_cfg,
            dataset=dataset_cfg,
            model=model_cfg,
            inference=inference_cfg,
            base_dir=base_dir,
        )

    @property
    def F_START_MHZ(self):
        return self.APP_CONFIG.filter.f_start_mhz

    @property
    def F_STOP_MHZ(self):
        return self.APP_CONFIG.filter.f_stop_mhz

    @property
    def BATCH_SIZE(self):
        return self.APP_CONFIG.inference.batch_size

    @property
    def BASE_DATASET_SIZE(self):
        return self.APP_CONFIG.dataset.size

    @property
    def FILTER_NAME(self):
        return  self.APP_CONFIG.filter.name

    @property
    def ENV_DEFAULT_FILTER_PATH(self):
        return str(self.APP_CONFIG.base_dir)

    @property
    def ENV_ORIGIN_DATA_PATH(self):
        return str(self.APP_CONFIG.base_dir / self.APP_CONFIG.paths.origins_data)

    @property
    def ENV_DATASET_PATH(self):
        return str(self.APP_CONFIG.base_dir / self.APP_CONFIG.paths.datasets_data)

    @property
    def ENV_TUNE_DATASET_PATH(self):
        return str(self.APP_CONFIG.base_dir / self.APP_CONFIG.paths.tune_data)

    @property
    def ENV_SAVED_MODELS_PATH(self):
        return str(self.APP_CONFIG.base_dir / self.APP_CONFIG.paths.saved_models)

    @property
    def MODEL_CHECKPOINT_PATH(self):
        return str(self.APP_CONFIG.base_dir / self.APP_CONFIG.paths.saved_models / self.APP_CONFIG.model.checkpoint)