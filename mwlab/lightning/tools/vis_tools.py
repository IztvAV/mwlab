"""
vis_tools.py
============

Утилиты **визуализации** S‑параметров для моделей MWLab + PyTorch Lightning.

Главная функция
---------------

plot_sparams_compare(...)
    • выбирает N случайных приборов из train/val/test/predict,
    • выводит на одном графике несколько заданных S‑параметров
      (истина — цветные линии, предсказание — черные точки).

Файл *НЕ* зависит от Lightning‑классов (используется только API trainer),
поэтому может применяться и вне тренировки.
"""

from __future__ import annotations
import random, itertools
from typing import List, Tuple, Literal

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from mwlab.io.touchstone import TouchstoneData
from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.lightning.touchstone_ldm import TouchstoneLDataModule


# ───────────────────────────────────────── внутренние функции
def _to_ts(obj, codec: TouchstoneCodec, meta=None) -> TouchstoneData:
    """
    Приводит произвольный объект из predict‑выхода к TouchstoneData.

    • TouchstoneData      → возврат как есть
    • Tensor (+ meta)     → codec.decode(tensor, meta)
    """
    if isinstance(obj, TouchstoneData):
        return obj
    if isinstance(obj, torch.Tensor):
        return codec.decode(obj, meta)
    raise TypeError(f"Неизвестный тип: {type(obj)}")


def _draw(ax, true_net, pred_net,
          pairs: List[Tuple[int, int]],
          *,
          unit_kind: str = "dB"):
    """
    Рисует S‑параметры **на общих осях**:

        • true_net  – цветные сплошные линии
        • pred_net  – чёрные маркеры «•»
    """
    # — частотная ось --------------------------------------------------
    # Берём масштаб из true‑сети (GHz / MHz / …), чтобы сохранить единицы.
    f_scaled = true_net.frequency.f_scaled
    f_unit   = true_net.frequency.unit or "Hz"

    # — палитра для разных пар портов ----------------------------------
    colors = itertools.cycle(plt.get_cmap("tab10").colors)

    # — для каждой пары (m,n) ------------------------------------------
    for (m, n), col in zip(pairs, colors):
        i, j = m - 1, n - 1         # перевод в 0‑based

        if unit_kind == "dB":
            y_t = true_net.s_db[:,  i, j]
            y_p = pred_net.s_db[:,  i, j]
        elif unit_kind == "mag":
            y_t = true_net.s_mag[:, i, j]
            y_p = pred_net.s_mag[:, i, j]
        elif unit_kind == "deg":
            y_t = true_net.s_deg[:, i, j]
            y_p = pred_net.s_deg[:, i, j]
        else:
            raise ValueError("unit_kind должен быть 'dB' | 'mag' | 'deg'")

        ax.plot(f_scaled, y_t, color=col, lw=1.4, label=f"S{m}{n} true")
        ax.plot(f_scaled, y_p, color="k", ls=":", ms=4, label=f"S{m}{n} pred")

    ax.set_xlabel(f"Частота ({f_unit})")
    ax.set_ylabel(unit_kind)
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)


# ───────────────────────────────────────── публичная функция
def plot_sparams_compare(
    trainer,
    pl_module,
    dm: TouchstoneLDataModule,
    *,
    split: Literal["train", "val", "test", "predict"] = "test",
    n_samples: int = 4,
    pairs: List[Tuple[int, int]] = [(1, 1), (2, 1)],
    unit: Literal["dB", "mag", "deg"] = "dB",
    rng_seed: int = 42,
):
    """
    Строит сравнение S‑параметров *истина / предсказание*.

    Parameters
    ----------
    split      : какой набор данных брать ('train'|'val'|'test'|'predict')
    n_samples  : число случайно выбранных приборов для визуализации
    pairs      : список портовых пар; индексы **с единицы** (S11→(1,1))
    unit       : 'dB' | 'mag' | 'deg' – что рисовать по оси Y
    """
    # 1) DataLoader, содержащий meta (нужно для codec.decode)
    if split == "predict":
        loader = dm.predict_dataloader()
    else:
        dm.setup("fit")
        loader = dm.get_dataloader(split, meta=True, shuffle=False)

    # 2) Получаем batched‑предсказания
    batch_preds = trainer.predict(pl_module, dataloaders=loader)

    preds, trues = [], []
    for (x, y, meta), bp in zip(loader, batch_preds):
        for k in range(len(bp)):
            preds.append(_to_ts(bp[k], dm.codec, meta[k]))
            trues.append(_to_ts(y[k],    dm.codec, meta[k]))

    # 3) случайно выбираем n_samples
    random.seed(rng_seed)
    idxs = random.sample(range(len(preds)), k=min(n_samples, len(preds)))

    # 4) Книга‑фигура из N осей
    fig, axes = plt.subplots(len(idxs), 1,
                             figsize=(7, 4 * len(idxs)),
                             squeeze=False)

    for row, idx in enumerate(idxs):
        _draw(axes[row][0],
              trues[idx].network,
              preds[idx].network,
              pairs,
              unit_kind=unit)

    fig.suptitle(f"{split.upper()}: {len(idxs)} samples")
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])
    return fig
