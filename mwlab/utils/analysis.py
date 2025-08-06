# mwlab/utils/analysis.py
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Iterable, Any, Dict, Optional, List, Literal

from mwlab import TouchstoneDataset
import skrf as rf


class TouchstoneDatasetAnalyzer:
    """
    Анализатор TouchstoneDataset: параметры + S‑матрицы.

    * dataset[idx][0] → dict с параметрами (float | nan)
    * dataset[idx][1] → skrf.Network

    Частотная сетка и единица измерения берутся из первого файла
    (`net.frequency.f` и `net.frequency.unit`) и проверяются
    на идентичность во всех остальных файлах.
    """

    # ------------------------- init -------------------------
    def __init__(self, dataset: TouchstoneDataset, eps_db: float = 1e-12):
        self.ds = dataset
        self._eps_db = eps_db

        # кеш для ускорения
        self._cache_params: Dict[int, Dict[str, float]] = {}
        self._cache_s: Dict[int, np.ndarray] = {}

        # итоговые представления
        self._params_df: Optional[pd.DataFrame] = None
        self._s_xarray: Optional[xr.DataArray] = None

        # частота и единица измерения
        self._freq_hz: Optional[np.ndarray] = None          # частоты в Hz
        self._freq_display: Optional[np.ndarray] = None     # масштабированные
        self._freq_unit: str = "Hz"                         # 'Hz' | 'kHz' …

    # ========================================================
    #                       ПАРАМЕТРЫ
    # ========================================================

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

    # mwlab/utils/analysis.py

    def summarize_params(
        self,
        *,
        include_numeric: bool = True,
        include_categorical: bool = False,
        coerce_numeric_strings: bool = False,
        drop_keys: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        """
        Сводная статистика по параметрам (X-колонкам).

        Параметры
        ---------
        include_numeric : bool, default True
            Включать ли числовые колонки. Метрики: mean/std/min/max/nan_count/is_constant.
        include_categorical : bool, default False
            Включать ли категориальные (строковые/объектные) колонки.
            Метрики: nunique/top/top_freq/nan_count/is_constant.
        coerce_numeric_strings : bool, default False
            Если True, пытаемся привести строковые столбцы к числам (pd.to_numeric, errors='coerce').
            Те, что успешно привелись (и не состоят полностью из NaN) → считаем числовыми.
        drop_keys : Iterable[str] | None
            Список колонок, которые нужно исключить перед подсчётом статистики
            (например, ['__id','__path','path','topo_name']).

        Возврат
        -------
        pd.DataFrame:
            Индекс строк: ['mean','std','min','max','nan_count','is_constant'] для числовых,
            и ['nunique','top','top_freq','nan_count','is_constant'] для категориальных.
            Колонки — выбранные ключи параметров.
        """
        df = self.get_params_df().copy()

        # опционально выкидываем служебные ключи
        if drop_keys:
            to_drop = [k for k in drop_keys if k in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)

        # --- Разделяем на numeric / non-numeric ---
        if coerce_numeric_strings:
            df_num_try = df.apply(pd.to_numeric, errors="coerce")
            numeric_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df_num_try[c].dtype)
                and not df_num_try[c].isna().all()
            ]
            df_num = df_num_try[numeric_cols]
        else:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            df_num = df[numeric_cols]

        obj_cols = [c for c in df.columns if c not in numeric_cols]

        # --- Блок: числовые метрики ---
        summary_num = pd.DataFrame()
        if include_numeric and numeric_cols:
            summary_num = pd.DataFrame({
                "mean":        df_num.mean(numeric_only=True),
                "std":         df_num.std(numeric_only=True),
                "min":         df_num.min(numeric_only=True),
                "max":         df_num.max(numeric_only=True),
                "nan_count":   df_num.isna().sum(),
                "is_constant": df_num.nunique(dropna=True) <= 1,
            }).T

        # --- Блок: категориальные метрики ---
        summary_cat = pd.DataFrame()
        if include_categorical and obj_cols:
            stats: Dict[str, Dict[str, Any]] = {}
            for col in obj_cols:
                s = df[col].astype("object")
                vc = s.value_counts(dropna=True)
                top_val = (vc.index[0] if not vc.empty else None)
                top_freq = (int(vc.iloc[0]) if not vc.empty else 0)
                stats[col] = {
                    "nunique":     int(s.nunique(dropna=True)),
                    "top":         top_val,
                    "top_freq":    top_freq,
                    "nan_count":   int(s.isna().sum()),
                    "is_constant": bool(s.nunique(dropna=True) <= 1),
                }
            if stats:
                summary_cat = pd.DataFrame(stats)

        # --- Возврат: по выбранным блокам ---
        if summary_num.empty and summary_cat.empty:
            warnings.warn(
                "summarize_params: результат пуст (проверьте include_numeric/include_categorical "
                "или drop_keys/coerce_numeric_strings).",
                RuntimeWarning,
                stacklevel=2,
            )
            return pd.DataFrame()

        if summary_num.empty:
            return summary_cat
        if summary_cat.empty:
            return summary_num
        return pd.concat([summary_num, summary_cat], axis=1, sort=False)

    def get_varying_keys(
        self,
        *,
        include_numeric: bool = True,
        include_categorical: bool = False,
        coerce_numeric_strings: bool = False,
        drop_keys: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Ключи, которые реально варьируются (is_constant == False)
        в выбранном подмножестве столбцов.
        """
        summary = self.summarize_params(
            include_numeric=include_numeric,
            include_categorical=include_categorical,
            coerce_numeric_strings=coerce_numeric_strings,
            drop_keys=drop_keys,
        )
        if "is_constant" not in summary.index or summary.empty:
            return []
        mask = summary.loc["is_constant"].astype(bool)
        return summary.columns[~mask].tolist()


    # ---------------------- графики параметров ----------------------

    def plot_param_distributions(self,
                                 keys: Optional[List[str]] = None,
                                 bins: int = 30):
        df = self.get_params_df()
        keys = keys or df.columns.tolist()
        if not keys:
            print("[⚠] Нет параметров для отображения.")
            return None

        n = len(keys)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flat if n > 1 else [axes]

        for ax, key in zip(axes, keys):
            sns.histplot(df[key].dropna(), bins=bins, ax=ax)
            ax.set_title(key)

        plt.tight_layout()
        return fig

    # ========================================================
    #                        S‑МАТРИЦЫ
    # ========================================================

    def _assemble_s_xarray(self) -> xr.DataArray:
        """
        Собирает единый `xr.DataArray`:

        dims:   sample • real_imag • freq • port_out • port_in
        coords: sample=0..N‑1,
                real_imag=['real','imag'],
                freq – частоты в единицах, заданных в Touchstone‑файле
                       (`net.frequency.unit`).
        """
        if self._s_xarray is not None:
            return self._s_xarray

        arrs: List[np.ndarray] = []
        ref_freq_hz: Optional[np.ndarray] = None
        ref_unit: Optional[str] = None

        # соответствие единиц → масштаб
        scale = {'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}
        disp   = {'hz': 'Hz', 'khz': 'kHz', 'mhz': 'MHz', 'ghz': 'GHz'}

        for idx in range(len(self.ds)):
            # ---------- читаем файл ----------
            if idx not in self._cache_s:
                _, net = self.ds[idx]
                if not isinstance(net, rf.Network):
                    raise TypeError("dataset[idx][1] должен быть skrf.Network")

                # ---------- проверяем частоты ----------
                if ref_freq_hz is None:
                    ref_freq_hz = net.frequency.f                         # Hz
                    ref_unit = net.frequency.unit.lower()                 # 'hz'…
                else:
                    if not np.allclose(net.frequency.f, ref_freq_hz,
                                       rtol=0, atol=1e-3):
                        raise ValueError(
                            f"Файл №{idx} имеет другую частотную сетку")
                    if net.frequency.unit.lower() != ref_unit:
                        raise ValueError(
                            f"Файл №{idx} имеет другую единицу частоты "
                            f"({net.frequency.unit})")

                # ---------- кешируем S ----------
                s = net.s.astype(np.complex64)                            # (F,P,P)
                self._cache_s[idx] = np.stack(
                    [s.real.astype(np.float32), s.imag.astype(np.float32)],
                    axis=0)                                               # (2,F,P,P)

            arrs.append(self._cache_s[idx])

        if not arrs:
            raise ValueError("TouchstoneDatasetAnalyzer: датасет пуст")

        data = np.stack(arrs, axis=0)             # (S,2,F,P,P)
        _, _, F, P, _ = data.shape

        # ---------- единица измерения ----------
        self._freq_hz = ref_freq_hz
        self._freq_unit = disp.get(ref_unit, ref_unit.upper())
        factor = scale.get(ref_unit, 1.0)
        self._freq_display = self._freq_hz / factor

        # ---------- DataArray ----------
        self._s_xarray = xr.DataArray(
            data,
            dims=('sample', 'real_imag', 'freq', 'port_out', 'port_in'),
            coords={
                'sample':    np.arange(len(arrs)),
                'real_imag': ['real', 'imag'],
                'freq':      self._freq_display,
                'port_out':  np.arange(P),
                'port_in':   np.arange(P),
            }
        )
        return self._s_xarray

    # --------------- представления (|S|, dB, φ) ---------------

    def _get_s_metric(self,
                      metric: Literal['db', 'mag', 'deg'] = 'db'
                      ) -> xr.DataArray:
        da = self._assemble_s_xarray()

        if metric == 'mag':
            return np.sqrt(
                da.sel(real_imag='real') ** 2 +
                da.sel(real_imag='imag') ** 2
            )

        if metric == 'db':
            mag = np.sqrt(
                da.sel(real_imag='real') ** 2 +
                da.sel(real_imag='imag') ** 2
            )
            return 20 * np.log10(mag + self._eps_db)

        if metric == 'deg':
            phase = np.arctan2(
                da.sel(real_imag='imag'),
                da.sel(real_imag='real')
            )
            unwrapped = np.unwrap(phase, axis=1) * 180 / np.pi
            coords = {k: v for k, v in da.coords.items()
                      if k != 'real_imag'}
            return xr.DataArray(
                unwrapped,
                dims=('sample', 'freq', 'port_out', 'port_in'),
                coords=coords
            )

        raise ValueError(f"Неподдерживаемая метрика: {metric}")

    # ------------------- статистика -------------------

    def summarize_s(self,
                    metric: Literal['db', 'mag', 'deg'] = 'db'
                    ) -> xr.Dataset:
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
                out[(po, pi)] = {
                    k: ds[k].sel(port_out=po, port_in=pi).values
                    for k in ('mean', 'std', 'min', 'max')
                }
        return out

    # -------------------------- графики --------------------------

    def plot_s_stats(self,
                     port_out: int = 1,
                     port_in: int = 1,
                     metric: Literal['db', 'mag', 'deg'] = 'db',
                     stats: List[str] = ('mean', 'std', 'min', 'max')):
        """
        График mean/std/min/max для выбранного порта (нумерация 1‑based).
        """
        ds = self.summarize_s(metric)
        freq = ds['mean']['freq'].values

        fig, ax = plt.subplots(figsize=(8, 4))

        # 1‑based → 0‑based
        po = port_out - 1
        pi = port_in - 1

        if 'mean' in stats:
            sns.lineplot(x=freq,
                         y=ds['mean'].sel(port_out=po, port_in=pi),
                         ax=ax, label='mean')

        if 'std' in stats:
            m = ds['mean'].sel(port_out=po, port_in=pi)
            s = ds['std'].sel(port_out=po, port_in=pi)
            ax.fill_between(freq, m - s, m + s, alpha=0.3, label='±1σ')

        for nm, style in (('min', '--'), ('max', '--')):
            if nm in stats:
                sns.lineplot(x=freq,
                             y=ds[nm].sel(port_out=po, port_in=pi),
                             ax=ax, linestyle=style, label=nm)

        unit_label = {'db': 'dB', 'mag': '|S|', 'deg': 'deg'}[metric]
        ax.set_xlabel(f'Frequency ({self._freq_unit})')
        ax.set_ylabel(f'S{po+1}{pi+1} ({unit_label})')
        ax.legend()
        plt.tight_layout()
        return fig

    def summarize_s_components(self) -> pd.DataFrame:
        """
        Вычисляет сводную статистику (mean, std, min, max, nan_count, is_constant)
        по всем компонентам S-параметров: Sij.real и Sij.imag,
        усреднённо по всем сэмплам и всем частотам.

        Возвращает:
            pd.DataFrame со строками: ['mean', 'std', ...]
            и колонками: ['S11.real', 'S11.imag', 'S12.real', ...]
        """
        da = self._assemble_s_xarray()  # (S, 2, F, P, P)

        stat = {}
        for po in da.coords['port_out'].values:
            for pi in da.coords['port_in'].values:
                for ri, label in enumerate(['real', 'imag']):
                    arr = da.sel(real_imag=label, port_out=po, port_in=pi).values  # (S, F)
                    arr_flat = arr.reshape(-1)

                    name = f"S{po + 1}{pi + 1}.{label}"
                    stat[name] = {
                        "mean": np.nanmean(arr_flat),
                        "std": np.nanstd(arr_flat),
                        "min": np.nanmin(arr_flat),
                        "max": np.nanmax(arr_flat),
                        "nan_count": np.isnan(arr_flat).sum(),
                        "is_constant": np.nanstd(arr_flat) < 1e-12,
                    }

        return pd.DataFrame(stat)

    # ========================================================
    #                        ЭКСПОРТ
    # ========================================================

    def export_params_csv(self, path: str):
        df = self.get_params_df()
        if df.empty:
            print(f"[⚠] Пустой DataFrame — не экспортируем в {path}")
            return
        df.to_csv(path, index=False)

    def export_s_netcdf(self, path: str):
        """Сохраняет статистику S‑параметров (mean/std/min/max) в NetCDF."""
        ds = self.summarize_s()
        ds.attrs['frequency_unit'] = self._freq_unit
        ds.to_netcdf(path)


