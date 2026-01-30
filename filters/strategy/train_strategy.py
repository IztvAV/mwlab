from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Any, Optional, Dict, Tuple
from filters.mwfilter_lightning import MWFilterBaseLMWithMetrics
import lightning as L
import re
import torch.nn as nn



class TrainingStrategy(Protocol):
    def fit(self, cfg: Any) -> Dict[str, Any]:
        ...


@dataclass
class BaseStrategyConfig:
    """
    Контейнер для уже созданных объектов:
      - datamodule (dm)
      - lightning module (lit_model)
      - trainer_template: тренер, который ты сам настроил (accelerator/log_every/callbacks/и т.д.)

    Стратегии будут:
      - запускать fit() один или несколько раз
      - при необходимости менять dm.batch_size
      - создавать новый trainer на каждую стадию (чтобы callbacks не тащили состояние)
    """
    dm: Any
    lit_model: L.LightningModule
    trainer_builder: Any
    trainer_builder_args: dict
    work_model: Any

    def set_batch_size(self, batch_size: int) -> None:
        if not hasattr(self.dm, "batch_size"):
            raise AttributeError("Datamodule has no attribute 'batch_size'.")
        self.dm.batch_size = int(batch_size)

    def new_trainer(self) -> Tuple[L.Trainer, Optional[L.Callback]]:
        """
        Создаёт новый Trainer на основе trainer_template, но с новым списком callbacks.
        Это гарантирует независимое состояние EarlyStopping/ModelCheckpoint для каждой стадии.
        """
        t = self.trainer_builder(**self.trainer_builder_args)
        return t



@dataclass
class StandardStrategy(TrainingStrategy):
    batch_size: int

    def fit(self, cfg) -> Dict[str, Any]:
        cfg.set_batch_size(self.batch_size)
        trainer, ckpt = cfg.new_trainer()
        trainer.fit(cfg.lit_model, datamodule=cfg.dm)
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        return {
            "status": "ok",
            "strategy": "standard",
            "best_ckpt": best_ckpt_path,
        }


@dataclass
class TwoStageBatchSizeStrategy(TrainingStrategy):
    small_bs: int
    large_bs: int

    def _rename_filename_with_new_batch_size_and_stage(self, filename:str|os.PathLike, stage:str, batch_size:int):
        pc = list(os.path.split(filename))
        pc[-1] = f"{stage}-{pc[-1]}"
        new_filename = os.path.join(*tuple(pc))
        os.replace(filename, new_filename)
        new_new_filename = re.sub(
            r"batch_size=\d+",
            f"batch_size={batch_size}",
            str(new_filename)
        )
        os.replace(new_filename, new_new_filename)
        return new_new_filename

    def fit(self, cfg) -> Dict[str, Any]:
        # stage 1
        cfg.set_batch_size(self.small_bs)
        trainer1 = cfg.new_trainer()
        trainer1.fit(cfg.lit_model, datamodule=cfg.dm)
        best1 = trainer1.checkpoint_callback.best_model_path
        best1 = self._rename_filename_with_new_batch_size_and_stage(best1, "stage1", cfg.dm.batch_size)

        # stage 2 (эквивалент статье: реальный batch_size)
        cfg.set_batch_size(self.large_bs)
        trainer2 = cfg.new_trainer()

        small_batch_model = MWFilterBaseLMWithMetrics.load_from_checkpoint(
            work_model=cfg.work_model,
            checkpoint_path=best1,
            model=cfg.lit_model.model
        ).to(cfg.lit_model.device)

        trainer2.fit(small_batch_model, datamodule=cfg.dm)
        best2 = trainer2.checkpoint_callback.best_model_path
        best2 = self._rename_filename_with_new_batch_size_and_stage(best2, "stage2", cfg.dm.batch_size)

        return {
            "status": "ok",
            "strategy": "two_stage_small_large",
            "best_small": best1,
            "best_large": best2,
        }
