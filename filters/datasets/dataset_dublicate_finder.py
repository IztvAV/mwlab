import os
import math
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class DuplicateDetectionConfig:
    enabled: bool = True

    # Количество знаков после запятой для грубой сигнатуры.
    # Чем меньше число, тем больше кандидатов и тем мягче предварительный фильтр.
    signature_decimals: int = 5

    # Порог для RMS по вектору [Re, Im]
    rms_threshold: float = 1e-4

    # Порог для максимального абсолютного расхождения по вектору [Re, Im]
    max_threshold: float = 5e-4

    # Сколько пар дубликатов рисовать максимум
    max_plots: int = 10

    # Сохранять графики
    save_plots: bool = True

    # Поддиректория для графиков/статистики
    report_subdir: str = "duplicate_report"

    # Сколько примеров хранить в каждом бакете.
    # Если опасаешься очень больших бакетов, можно ограничить.
    max_bucket_size: Optional[int] = None

    # Если True, то хранить все признаки и делать финальную статистику по ним.
    store_all_features: bool = True


class DuplicateStatsCollector:
    """
    Сборщик статистики по дубликатам в потоковом режиме во время генерации датасета.
    """

    def __init__(self, base_dir: str, config: DuplicateDetectionConfig):
        self.config = config
        self.base_dir = base_dir
        self.report_dir = os.path.join(base_dir, config.report_subdir)
        os.makedirs(self.report_dir, exist_ok=True)

        # bucket_key -> list[record]
        self._buckets: Dict[bytes, List[Dict[str, Any]]] = defaultdict(list)

        self.total_samples = 0
        self.total_candidate_checks = 0
        self.total_duplicates = 0

        self.duplicate_pairs: List[Dict[str, Any]] = []
        self.sample_infos: List[Dict[str, Any]] = []

    @staticmethod
    def _to_numpy_complex_s(ts) -> np.ndarray:
        """
        Ожидается формат skrf-like:
        ts.s.shape = (nfreq, nports, nports)
        """
        s = ts.s
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        return np.asarray(s)

    @staticmethod
    def build_feature_vector_from_ts(ts) -> np.ndarray:
        """
        Преобразует комплексные S-параметры в 1D-вектор:
        [Re(flatten(S)), Im(flatten(S))]
        или точнее interleave по последней размерности:
        [Re(S11(f1)), Im(S11(f1)), Re(S12(f1)), Im(S12(f1)), ...]
        """
        s = DuplicateStatsCollector._to_numpy_complex_s(ts)  # (nf, p, p)
        # flatten complex entries over all freq and all matrix entries
        s_flat = s.reshape(-1)
        feat = np.empty(s_flat.size * 2, dtype=np.float32)
        feat[0::2] = s_flat.real.astype(np.float32)
        feat[1::2] = s_flat.imag.astype(np.float32)
        return feat

    def build_signature(self, feat: np.ndarray) -> bytes:
        rounded = np.round(feat, decimals=self.config.signature_decimals)
        return rounded.tobytes()

    @staticmethod
    def compare_features(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
        diff = x - y
        rms = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(np.abs(diff)))
        norm_x = float(np.sqrt(np.mean(x * x)))
        rel_rms = float(rms / (norm_x + eps))
        return {
            "rms": rms,
            "max_abs": max_abs,
            "rel_rms": rel_rms,
        }

    def is_duplicate(self, metrics: Dict[str, float]) -> bool:
        return (
            metrics["rms"] <= self.config.rms_threshold
            and metrics["max_abs"] <= self.config.max_threshold
        )

    def add_sample(
        self,
        idx: int,
        ts,
        params: np.ndarray,
        meta: Dict[str, Any],
    ):
        if not self.config.enabled:
            return

        feat = self.build_feature_vector_from_ts(ts)
        signature = self.build_signature(feat)

        record = {
            "idx": idx,
            "feature": feat,
            "params": params.astype(np.float32, copy=False),
            "meta": meta,
            "ts": ts,
        }

        bucket = self._buckets[signature]

        for prev in bucket:
            self.total_candidate_checks += 1
            metrics = self.compare_features(feat, prev["feature"])
            if self.is_duplicate(metrics):
                self.total_duplicates += 1
                pair_info = {
                    "idx_a": prev["idx"],
                    "idx_b": idx,
                    "metrics": metrics,
                    "params_a": prev["params"].tolist(),
                    "params_b": record["params"].tolist(),
                    "meta_a": prev["meta"],
                    "meta_b": meta,
                    "ts_a": prev["ts"],
                    "ts_b": ts,
                }
                self.duplicate_pairs.append(pair_info)

        if self.config.max_bucket_size is None or len(bucket) < self.config.max_bucket_size:
            bucket.append(record)

        self.total_samples += 1

        if self.config.store_all_features:
            self.sample_infos.append(
                {
                    "idx": idx,
                    "meta": meta,
                    "params": params.tolist(),
                }
            )

    def summary(self) -> Dict[str, Any]:
        unique_with_duplicates = set()
        for pair in self.duplicate_pairs:
            unique_with_duplicates.add(pair["idx_a"])
            unique_with_duplicates.add(pair["idx_b"])

        return {
            "total_samples": self.total_samples,
            "total_candidate_checks": self.total_candidate_checks,
            "total_duplicate_pairs": self.total_duplicates,
            "samples_in_duplicate_pairs": len(unique_with_duplicates),
            "share_samples_in_duplicate_pairs": (
                len(unique_with_duplicates) / self.total_samples if self.total_samples > 0 else 0.0
            ),
            "num_buckets": len(self._buckets),
            "config": asdict(self.config),
        }

    def save_summary(self):
        path = os.path.join(self.report_dir, "duplicate_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)

    def save_pairs_table(self):
        path = os.path.join(self.report_dir, "duplicate_pairs.json")
        serializable_pairs = []
        for pair in self.duplicate_pairs:
            serializable_pairs.append(
                {
                    "idx_a": pair["idx_a"],
                    "idx_b": pair["idx_b"],
                    "metrics": pair["metrics"],
                    "params_a": pair["params_a"],
                    "params_b": pair["params_b"],
                    "meta_a": pair["meta_a"],
                    "meta_b": pair["meta_b"],
                }
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable_pairs, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_frequency(ts) -> np.ndarray:
        """
        Пытаемся достать ось частоты.
        Для scikit-rf Touchstone/Network это обычно:
        ts.f либо ts.frequency.f
        """
        if hasattr(ts, "f"):
            f = ts.f
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
            return np.asarray(f)

        if hasattr(ts, "frequency") and hasattr(ts.frequency, "f"):
            f = ts.frequency.f
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
            return np.asarray(f)

        raise AttributeError("Не удалось извлечь частотную ось из ts")

    @staticmethod
    def _get_s(ts) -> np.ndarray:
        s = ts.s
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        return np.asarray(s)

    def plot_duplicate_pair(self, pair: Dict[str, Any], pair_number: int):
        ts_a = pair["ts_a"]
        ts_b = pair["ts_b"]

        f = self._extract_frequency(ts_a)
        s_a = self._get_s(ts_a)
        s_b = self._get_s(ts_b)

        if s_a.shape != s_b.shape:
            raise ValueError("Формы S-параметров у пары не совпадают")

        nfreq, nports, _ = s_a.shape

        fig, axes = plt.subplots(3, nports, nports, figsize=(4 * nports, 9 * nports))
        if nports == 1:
            axes = np.array([[axes[0]], [axes[1]], [axes[2]]], dtype=object)

        def s_db(x):
            return 20.0 * np.log10(np.maximum(np.abs(x), 1e-15))

        titles = []
        for i in range(nports):
            for j in range(nports):
                titles.append((i, j))

        for i in range(nports):
            for j in range(nports):
                # dB
                ax = axes[0, i, j]
                ax.plot(f, s_db(s_a[:, i, j]), label=f"A S{i+1}{j+1}")
                ax.plot(f, s_db(s_b[:, i, j]), "--", label=f"B S{i+1}{j+1}")
                ax.set_title(f"|S{i+1}{j+1}|, dB")
                ax.grid(True)
                if i == 0 and j == 0:
                    ax.legend()

                # Re
                ax = axes[1, i, j]
                ax.plot(f, s_a[:, i, j].real, label=f"A Re(S{i+1}{j+1})")
                ax.plot(f, s_b[:, i, j].real, "--", label=f"B Re(S{i+1}{j+1})")
                ax.set_title(f"Re(S{i+1}{j+1})")
                ax.grid(True)
                if i == 0 and j == 0:
                    ax.legend()

                # Im
                ax = axes[2, i, j]
                ax.plot(f, s_a[:, i, j].imag, label=f"A Im(S{i+1}{j+1})")
                ax.plot(f, s_b[:, i, j].imag, "--", label=f"B Im(S{i+1}{j+1})")
                ax.set_title(f"Im(S{i+1}{j+1})")
                ax.grid(True)
                if i == 0 and j == 0:
                    ax.legend()

        fig.suptitle(
            f"Duplicate pair #{pair_number}: idx {pair['idx_a']} vs {pair['idx_b']}\n"
            f"RMS={pair['metrics']['rms']:.3e}, "
            f"MAX={pair['metrics']['max_abs']:.3e}, "
            f"REL_RMS={pair['metrics']['rel_rms']:.3e}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = os.path.join(self.report_dir, f"duplicate_pair_{pair_number:03d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_plots(self):
        if not self.config.save_plots:
            return

        n = min(self.config.max_plots, len(self.duplicate_pairs))
        for i in range(n):
            self.plot_duplicate_pair(self.duplicate_pairs[i], i + 1)

    def finalize(self):
        self.save_summary()
        self.save_pairs_table()
        self.save_plots()
