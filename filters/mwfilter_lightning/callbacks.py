import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from dataclasses import dataclass


class SaflySaveCheckpoint(ModelCheckpoint):
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


@dataclass
class BaseCallbackSet:
    early_stopping: L.pytorch.callbacks.EarlyStopping
    save_checkpoint: SaflySaveCheckpoint

    @property
    def cb_list(self):
        return [self.early_stopping, self.save_checkpoint]