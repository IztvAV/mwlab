import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np

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
class MatrixSamplerConfig:
    self_coupling: float
    mainline_coupling: float
    cross_coupling: float
    parasitic_coupling: float
    absolute: bool


@dataclass
class PssSampler:
    a11: float
    a22: float
    b11: float
    b22: float

    def array(self):
        return np.array([self.a11, self.a22, self.b11, self.b22])


@dataclass
class DatasetSizeConfig:
    train: int
    infer: int

@dataclass
class DatasetConfig:
    matrix_sampler_delta: MatrixSamplerConfig
    pss_origin: PssSampler
    pss_sampler_delta: PssSampler
    size: DatasetSizeConfig  # раньше BASE_DATASET_SIZE
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
        dataset_section = data["dataset"]

        matrix_sampler_cfg = MatrixSamplerConfig(**dataset_section["matrix_sampler_delta"])
        pss_origin_cfg = PssSampler(**dataset_section["pss_origin"])
        pss_sampler_cfg = PssSampler(**dataset_section["pss_sampler_delta"])
        dataset_size_cfg = DatasetSizeConfig(**dataset_section['size'])

        dataset_cfg = DatasetConfig(
            size=dataset_size_cfg,
            sampler_type=dataset_section["sampler_type"],
            matrix_sampler_delta=matrix_sampler_cfg,
            pss_origin=pss_origin_cfg,
            pss_sampler_delta=pss_sampler_cfg,
        )

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

    # @property
    # def BASE_DATASET_SIZE(self):
    #     return self.APP_CONFIG.dataset.size

    @property
    def TRAIN_DATASET_SIZE(self):
        return self.APP_CONFIG.dataset.size.train

    @property
    def INFERENCE_DATASET_SIZE(self):
        return self.APP_CONFIG.dataset.size.infer

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