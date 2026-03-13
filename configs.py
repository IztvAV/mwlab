import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import yaml
from omegaconf import OmegaConf, DictConfig


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
class TrainConfig:
    strategy_type: str
    max_epochs: int


@dataclass
class ModelConfig:
    model_name: str # название модели
    codec: str # тип кодека
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
    train: TrainConfig
    model: ModelConfig
    inference: InferenceConfig

    base_dir: Path  # добавляем поле, чтобы потом легко получать ENV_* пути

# твои dataclass'ы:
# MetaConfig, FilterConfig, PathsConfig, MatrixSamplerConfig,
# PssSampler, DatasetSizeConfig, DatasetConfig,
# TrainConfig, ModelConfig, InferenceConfig, AppConfig


class Configs:
    def __init__(self, *config_paths: str | os.PathLike):
        """
        Можно передавать:
            Configs("base.yml")
            Configs("base.yml", "datasets/cm.yml", "models/cm.yml")
        """
        if not config_paths:
            raise ValueError("At least one config path must be provided")

        self.INFERENCE_DEVICE: str | None = None

        self._config_paths = [Path(p).resolve() for p in config_paths]
        self._merged_cfg: DictConfig = self.load_merged_configs(*self._config_paths)
        self.APP_CONFIG: AppConfig = self._build_app_config(self._merged_cfg, self._config_paths[0])

    @classmethod
    def from_files(cls, *config_paths: str | os.PathLike) -> "Configs":
        return cls(*config_paths)

    @classmethod
    def init_from_default(cls, default_path: str | os.PathLike) -> "Configs":
        """
        Оставляем совместимость со старой схемой.
        default_path может указывать на yaml, в котором лежит список/путь до нужных конфигов.
        """
        config_paths = cls._get_config_paths_from_default(default_path)
        return cls(*config_paths)

    @staticmethod
    def _get_config_paths_from_default(default_path: os.PathLike | str) -> list[str | os.PathLike]:
        """
        Вариант 1:
            default.yml содержит:
            paths:
              manifest: manifest.yml

        Вариант 2:
            default.yml содержит:
            configs:
              - conf/base.yml
              - conf/datasets/cm_default.yml
              - conf/models/cm_extract.yml
        """
        default_path = Path(default_path).resolve()

        if not default_path.exists():
            raise FileNotFoundError(f"Default config file not found: {default_path}")

        cfg = OmegaConf.load(default_path)
        data = OmegaConf.to_container(cfg, resolve=True)

        # новая схема: список файлов
        if "configs" in data:
            result = []
            for p in data["configs"]:
                p = Path(p)
                if not p.is_absolute():
                    p = (default_path.parent / p).resolve()
                result.append(p)
            return result

        # старая схема: один manifest
        if "paths" in data and "manifest" in data["paths"]:
            manifest_path = Path(data["paths"]["manifest"])
            if not manifest_path.is_absolute():
                manifest_path = (default_path.parent / manifest_path).resolve()
            return [manifest_path]

        raise ValueError(
            f"Unsupported default config format in {default_path}. "
            f"Expected either 'configs' list or 'paths.manifest'."
        )

    @staticmethod
    def load_merged_configs(*config_paths: str | os.PathLike) -> DictConfig:
        cfgs = []

        for p in config_paths:
            path = Path(p).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            cfgs.append(OmegaConf.load(path))

        merged = OmegaConf.merge(*cfgs)
        return merged

    @staticmethod
    def _build_app_config(cfg: DictConfig, first_config_path: Path) -> AppConfig:
        """
        first_config_path нужен для определения base_dir.
        Обычно это base.yml внутри директории фильтра.
        """
        data = OmegaConf.to_container(cfg, resolve=True)

        # базовая директория фильтра:
        # предполагаем, что первый конфиг лежит в:
        # FilterData/<filter_name>/conf/base.yml
        #
        # Тогда base_dir = FilterData/<filter_name>
        #
        # Если у тебя конфиги лежат прямо в корне фильтра, можно поменять логику на .parent
        if first_config_path.parent.name == "conf":
            base_dir = first_config_path.parent.parent
        else:
            base_dir = first_config_path.parent

        meta_cfg = MetaConfig(**data["meta"])
        filter_cfg = FilterConfig(**data["filter"])
        paths_cfg = PathsConfig(**data["paths"])

        dataset_section = data["dataset"]
        matrix_sampler_cfg = MatrixSamplerConfig(**dataset_section["matrix_sampler_delta"])
        pss_origin_cfg = PssSampler(**dataset_section["pss_origin"])
        pss_sampler_cfg = PssSampler(**dataset_section["pss_sampler_delta"])
        dataset_size_cfg = DatasetSizeConfig(**dataset_section["size"])

        dataset_cfg = DatasetConfig(
            size=dataset_size_cfg,
            sampler_type=dataset_section["sampler_type"],
            matrix_sampler_delta=matrix_sampler_cfg,
            pss_origin=pss_origin_cfg,
            pss_sampler_delta=pss_sampler_cfg,
        )

        train_cfg = TrainConfig(**data["train"])
        model_cfg = ModelConfig(**data["model"])
        inference_cfg = InferenceConfig(**data["inference"])

        return AppConfig(
            meta=meta_cfg,
            filter=filter_cfg,
            paths=paths_cfg,
            dataset=dataset_cfg,
            train=train_cfg,
            model=model_cfg,
            inference=inference_cfg,
            base_dir=base_dir,
        )

    @property
    def RAW_CONFIG(self) -> DictConfig:
        return self._merged_cfg

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
    def TRAIN_DATASET_SIZE(self):
        return self.APP_CONFIG.dataset.size.train

    @property
    def INFERENCE_DATASET_SIZE(self):
        return self.APP_CONFIG.dataset.size.infer

    @property
    def FILTER_NAME(self):
        return self.APP_CONFIG.filter.name

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
        return str(
            self.APP_CONFIG.base_dir
            / self.APP_CONFIG.paths.saved_models
            / self.APP_CONFIG.model.checkpoint
        )