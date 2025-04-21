# utils/analysis.py
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional, List, Literal, Dict, Any
#from mwlab import TouchstoneData
from mwlab import TouchstoneDataset
import skrf as rf


class TouchstoneDatasetAnalyzer:
    def __init__(self,
                 dataset: TouchstoneDataset,
                 common_freq: Optional[np.ndarray] = None):
        """
        Анализатор TouchstoneDataset: параметры + S‑матрицы.

        Предполагается, что `dataset[idx][1]` возвращает объект `skrf.Network`.

        Parameters
        ----------
        dataset : TouchstoneDataset
            Набор Touchstone-файлов.
        common_freq : array-like, optional
            Эталонная частотная сетка, если нужна проверка.
        """
        self.ds = dataset
        self.common_freq = np.asarray(common_freq) if common_freq is not None else None
        self._cache_params: Dict[int, Dict[str, float]] = {}
        self._cache_s: Dict[int, np.ndarray] = {}
        self._params_df: Optional[pd.DataFrame] = None
        self._s_xarray: Optional[xr.DataArray] = None

    # ---------------------- ПАРАМЕТРЫ ----------------------

    def get_params_df(self) -> pd.DataFrame:
        if self._params_df is None:
            records = []
            for idx in range(len(self.ds)):
                if idx not in self._cache_params:
                    x, _ = self.ds[idx]
                    self._cache_params[idx] = x
                records.append(self._cache_params[idx])
            self._params_df = pd.DataFrame(records)
        return self._params_df

    def summarize_params(self) -> pd.DataFrame:
        df = self.get_params_df()
        summary = pd.DataFrame({
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            'max': df.max(),
            'nan_count': df.isna().sum(),
            'is_constant': df.nunique(dropna=True) <= 1
        }).T
        return summary

    def get_varying_keys(self) -> List[str]:
        summary = self.summarize_params()
        if "is_constant" not in summary.index:
            return []
        mask = summary.loc["is_constant"].astype(bool)
        return summary.columns[~mask].tolist()

    def plot_param_distributions(self,
                                 keys: Optional[List[str]] = None,
                                 bins: int = 30):
        df = self.get_params_df()
        keys = keys or df.columns.tolist()
        n = len(keys)
        if n == 0:
            print("[⚠] Нет параметров для отображения.")
            return None
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flat if n > 1 else [axes]
        for ax, key in zip(axes, keys):
            sns.histplot(df[key].dropna(), bins=bins, ax=ax)
            ax.set_title(key)
        plt.tight_layout()
        return fig

    # ---------------------- S-МАТРИЦЫ ----------------------

    def _assemble_s_xarray(self) -> xr.DataArray:
        if self._s_xarray is None:
            arrs = []
            for idx in range(len(self.ds)):
                if idx not in self._cache_s:
                    _, net = self.ds[idx]
                    assert isinstance(net, rf.Network), "Ожидается skrf.Network"
                    s = net.s.astype(np.complex64)
                    arr = np.stack([s.real.astype(np.float32),
                                    s.imag.astype(np.float32)], axis=0)  # (2, F, P, P)
                    self._cache_s[idx] = arr
                arrs.append(self._cache_s[idx])
            if not arrs:
                raise ValueError("TouchstoneDatasetAnalyzer: датасет пуст")
            data = np.stack(arrs, axis=0)  # (S, 2, F, P, P)
            F = data.shape[2]
            P = data.shape[3]
            freqs = self.common_freq if self.common_freq is not None else np.arange(F)
            self._s_xarray = xr.DataArray(
                data,
                dims=('sample','real_imag','freq','port_out','port_in'),
                coords={
                    'sample':    np.arange(len(arrs)),
                    'real_imag': ['real','imag'],
                    'freq':      freqs,
                    'port_out':  np.arange(P),
                    'port_in':   np.arange(P),
                }
            )
        return self._s_xarray

    def _get_s_metric(self,
                      metric: Literal['db', 'mag', 'deg'] = 'db') -> xr.DataArray:
        da = self._assemble_s_xarray()
        if metric == 'db':
            mag = np.sqrt(
                da.sel(real_imag='real') ** 2 +
                da.sel(real_imag='imag') ** 2
            )
            return 20 * np.log10(mag)
        elif metric == 'mag':
            return np.sqrt(
                da.sel(real_imag='real') ** 2 +
                da.sel(real_imag='imag') ** 2
            )
        elif metric == 'deg':
            phase = np.arctan2(
                da.sel(real_imag='imag'),
                da.sel(real_imag='real')
            )
            unwrapped = np.unwrap(phase, axis=1) * 180 / np.pi
            coords = {k: v for k, v in da.coords.items() if k != 'real_imag'}
            return xr.DataArray(
                unwrapped,
                dims=('sample', 'freq', 'port_out', 'port_in'),
                coords=coords
            )
        else:
            raise ValueError(f"Неподдерживаемая метрика: {metric}")

    def summarize_s(self,
                    metric: Literal['db', 'mag', 'deg'] = 'db') -> xr.Dataset:
        da = self._get_s_metric(metric)
        return xr.Dataset({
            'mean': da.mean(dim='sample'),
            'std':  da.std(dim='sample'),
            'min':  da.min(dim='sample'),
            'max':  da.max(dim='sample'),
        })

    def get_s_normalization_stats(self,
                                  metric: Literal['db', 'mag', 'deg'] = 'db'
                                  ) -> Dict[Any, Dict[str, np.ndarray]]:
        ds = self.summarize_s(metric)
        out: Dict = {}
        for po in ds['port_out'].values:
            for pi in ds['port_in'].values:
                out[(po,pi)] = {
                    'mean': ds['mean'].sel(port_out=po, port_in=pi).values,
                    'std':  ds['std'].sel(port_out=po, port_in=pi).values,
                    'min':  ds['min'].sel(port_out=po, port_in=pi).values,
                    'max':  ds['max'].sel(port_out=po, port_in=pi).values,
                }
        return out

    def plot_s_stats(self,
                     port_out: int = 1,
                     port_in:  int = 1,
                     metric:   Literal['db', 'mag', 'deg'] = 'db',
                     stats:    List[str] = ['mean','std','min','max']):
        ds = self.summarize_s(metric)
        freq = ds['freq'].values
        fig, ax = plt.subplots(figsize=(8,4))

        port_out -= 1
        port_in -= 1

        if 'mean' in stats:
            sns.lineplot(x=freq,
                         y=ds['mean'].sel(port_out=port_out, port_in=port_in),
                         ax=ax, label='mean')

        if 'std' in stats:
            m = ds['mean'].sel(port_out=port_out, port_in=port_in)
            s = ds['std'].sel(port_out=port_out, port_in=port_in)
            ax.fill_between(freq, m-s, m+s, alpha=0.3, label='±1σ')

        for nm, style in [('min','--'), ('max','--')]:
            if nm in stats:
                sns.lineplot(x=freq,
                             y=ds[nm].sel(port_out=port_out, port_in=port_in),
                             ax=ax, linestyle=style, label=nm)

        unit_label = {'db': 'dB', 'mag': '|S|', 'deg': 'deg'}[metric]
        unit_freq = getattr(self.common_freq, "dtype", "Hz") if self.common_freq is not None else "Hz"
        ax.set_xlabel(f'Frequency ({unit_freq})')
        ax.set_ylabel(f'S{port_out+1}{port_in+1} ({unit_label})')
        ax.legend()
        plt.tight_layout()
        return fig

    # ---------------------- ЭКСПОРТ ----------------------
    def export_params_csv(self, path: str):
        df = self.get_params_df()
        if df.empty:
            print(f"[⚠] Пустой DataFrame — не экспортируем в {path}")
            return
        df.to_csv(path, index=False)

    def export_s_netcdf(self, path: str):
        self.summarize_s().to_netcdf(path)

