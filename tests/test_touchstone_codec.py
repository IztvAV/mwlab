# tests/test_touchstone_codec.py
"""
Набор юнит‑тестов для `TouchstoneCodec` — преобразователя между `TouchstoneData` и PyTorch‑тензорами.

Проверяем корректность работы кодека:

    ✓ Полный цикл encode → decode с различными компонентами:
        • real/imag
        • db/deg
        • magnitude (amp-only)
    ✓ Обработка NaN-параметров в X-векторе;
    ✓ Валидация и защита от некорректных названий каналов;
    ✓ Автоматический ресемплинг частот при encode();
    ✓ Ошибка при несоответствии частот и отключённом force_resample;
    ✓ Поддержка восстановления без meta (fallback к unit='Hz', z0=50Ω);
    ✓ Корректная реконструкция z0 (скаляр и векторная форма);
    ✓ Сохранение и восстановление системных метаданных:
        • единицы измерения (unit)
        • определение волн (s_def)
        • опорный импеданс (z0)
        • комментарии
    ✓ Поддержка сериализации и десериализации через pickle (dumps / loads);
    ✓ Автоматическая генерация codec через from_dataset().
"""


from __future__ import annotations

import numpy as np
import pytest
import skrf as rf
import torch
import math

from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.io.touchstone import TouchstoneData
from mwlab.datasets.touchstone_dataset import TouchstoneDataset


# ───────────────────────────────────────────────── helpers ──────────────────
def _make_touchstone(n_ports: int = 2, n_freq: int = 101) -> TouchstoneData:
    """
    Генерируем стабильные искусственные данные с заметной фазой,
    отличным от 50 Ω импедансом и user‑params.
    """
    rng = np.random.default_rng(0)
    f = np.linspace(1e9, 5e9, n_freq)              # 1…5 ГГц
    s = (rng.standard_normal((n_freq, n_ports, n_ports))
         + 1j * rng.standard_normal((n_freq, n_ports, n_ports))) * 0.05

    freq = rf.Frequency.from_f(f, unit="Hz")
    z0 = np.full(n_ports, 60.0)                    # нестандартный импеданс
    net = rf.Network(frequency=freq, s=s, z0=z0)
    net.comments = ["test network"]

    params = {"a": 1.0, "b": 2.0}
    return TouchstoneData(net, params=params)


def _assert_network_equal(a: rf.Network, b: rf.Network, *, atol=1e-8, rtol=1e-6):
    """Проверяем полное совпадение сетей и мета‑полей."""
    assert a.number_of_ports == b.number_of_ports
    assert a.frequency.unit == b.frequency.unit
    assert a.s_def == b.s_def
    np.testing.assert_allclose(a.z0, b.z0, atol=0, rtol=0)
    np.testing.assert_allclose(a.f, b.f, atol=0, rtol=1e-12)
    np.testing.assert_allclose(a.s, b.s, atol=atol, rtol=rtol)

def _group_delay_ref(net: rf.Network, i: int, j: int):
    """Возвращает эталон GD(i,j) shape (F,)."""
    return net.group_delay[:, i, j]


class _DummyTouchstoneDataset(TouchstoneDataset):
    def __init__(self):
        self._ts = _make_touchstone()
        self._len = 1

    def __len__(self): return self._len

    def __getitem__(self, idx): return self._ts.params, self._ts.network

# ───────────────────────────────────────────────── fixtures ──────────────────
@pytest.fixture(scope="module")
def ts_sample():
    return _make_touchstone()


@pytest.fixture(scope="module")
def freq_vec(ts_sample):
    return ts_sample.network.f


# ───────────────────────────────────────────────── tests ─────────────────────
def _roundtrip(codec: TouchstoneCodec, ts: TouchstoneData, *, atol=1e-8):
    x_t, y_t, meta = codec.encode(ts)
    ts_rec = codec.decode(y_t, meta)
    _assert_network_equal(ts.network, ts_rec.network, atol=atol)


def test_roundtrip_real_imag(ts_sample, freq_vec):
    y_ch = [f"S{i}_{j}.{p}" for p in ("real", "imag") for i in (1, 2) for j in (1, 2)]
    codec = TouchstoneCodec(x_keys=["a", "b"], y_channels=y_ch, freq_hz=freq_vec)
    _roundtrip(codec, ts_sample)

def test_roundtrip_real_imag_gd(ts_sample, freq_vec):
    """Проверяем, что добавление GD-каналов не ломает реконструкцию."""

    y_ch = [                           # real, imag, gd для 2-портов
        f"S{i}_{j}.{p}"
        for p in ("real", "imag", "gd")
        for i in (1, 2)
        for j in (1, 2)
    ]
    codec = TouchstoneCodec(x_keys=["a", "b"], y_channels=y_ch, freq_hz=freq_vec)

    # ----- encode
    x_t, y_t, meta = codec.encode(ts_sample)
    assert y_t.shape[0] == 3 * 4          # 3 компонента × 4 пары портов

    # ----- decode  →  сеть не должна менять S-матрицу
    ts_rec = codec.decode(y_t, meta)
    _assert_network_equal(ts_sample.network, ts_rec.network, atol=1e-8)

    # ----- sanity-check: строка GD реально совпадает с reference
    gd_ch_idx = codec.y_channels.index("S1_1.gd")
    gd_pred   = y_t[gd_ch_idx].numpy()
    gd_ref    = _group_delay_ref(ts_sample.network, 0, 0)
    np.testing.assert_allclose(gd_pred, gd_ref, atol=1e-12, rtol=1e-9)


def test_roundtrip_db_deg(ts_sample, freq_vec):
    y_ch = [f"S{i}_{j}.{p}" for p in ("db", "deg") for i in (1, 2) for j in (1, 2)]
    codec = TouchstoneCodec(x_keys=["a", "b"], y_channels=y_ch, freq_hz=freq_vec)
    _roundtrip(codec, ts_sample, atol=1e-6)


def test_amp_only_nan_fill(ts_sample, freq_vec):
    y_ch = [f"S{i}_{j}.mag" for i in (1, 2) for j in (1, 2)]
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=y_ch,
        freq_hz=freq_vec,
        nan_fill=0.0 + 0.0j,
    )

    _, y_t, meta = codec.encode(ts_sample)
    ts_rec = codec.decode(y_t, meta)

    s_rec = ts_rec.network.s
    s_ref = ts_sample.network.s
    np.testing.assert_allclose(np.abs(s_rec), np.abs(s_ref), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(s_rec.imag, 0.0, atol=1e-9)


@pytest.mark.parametrize("bad_tag", ["S1X.real", "S223.imag", "S11.foo", "S1_1.GDx"])
def test_parse_channel_validation(bad_tag):
    codec = TouchstoneCodec(x_keys=["a"], y_channels=["S1_1.real"], freq_hz=np.array([1.0]))
    with pytest.raises(ValueError):
        codec._parse_channel(bad_tag)

def test_parse_channel_gd_ok():
    codec = TouchstoneCodec(x_keys=["a"], y_channels=["S1_1.real"], freq_hz=np.array([1.0]))
    i, j, part = codec._parse_channel("S1_1.gd")
    assert (i, j, part) == (0, 0, "gd")

def test_resample_encode(ts_sample):
    freq_new = np.linspace(1e9, 5e9, 51)
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real", "S1_1.imag"],
        freq_hz=freq_new,
    )

    _, y_t, meta = codec.encode(ts_sample)
    assert y_t.shape[1] == len(freq_new)

    ts_rec = codec.decode(y_t, meta)
    np.testing.assert_allclose(ts_rec.network.f, freq_new, atol=0, rtol=1e-12)
    assert np.isfinite(ts_rec.network.s).all()

def test_force_resample_off_error(ts_sample):
    freq_shifted = ts_sample.network.f * 1.01  # нарочно сдвинули
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real", "S1_1.imag"],
        freq_hz=freq_shifted,
        force_resample=False,
    )

    with pytest.raises(ValueError, match="encode_s: несоответствие частотной сетки"):
        codec.encode(ts_sample)

def test_decode_without_meta(ts_sample, freq_vec):
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real", "S1_1.imag"],
        freq_hz=freq_vec,
    )
    _, y_t, _ = codec.encode(ts_sample)
    ts_dec = codec.decode(y_t, meta=None)

    # Проверим базовые свойства
    assert isinstance(ts_dec, TouchstoneData)
    assert ts_dec.network.z0.shape == (len(freq_vec), codec.n_ports)
    np.testing.assert_allclose(ts_dec.network.z0, 50, atol=1e-5)

def test_nan_in_params(freq_vec):
    ts = _make_touchstone()
    ts.params["unused"] = np.nan

    codec = TouchstoneCodec(
        x_keys=["a", "b", "unused"],
        y_channels=["S1_1.real", "S1_1.imag"],
        freq_hz=freq_vec,
    )
    x_t, _, _ = codec.encode(ts)
    assert torch.isnan(x_t[-1])  # последний параметр был NaN

def test_z0_scalar_and_vector_handling(freq_vec):
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real", "S1_1.imag"],
        freq_hz=freq_vec,
    )

    _, y_t, meta = codec.encode(_make_touchstone())
    meta["z0"] = 75.0
    ts1 = codec.decode(y_t, meta)
    assert np.allclose(ts1.network.z0, 75.0)

    meta["z0"] = np.array([75.0, 75.0])
    ts2 = codec.decode(y_t, meta)
    assert np.allclose(ts2.network.z0, 75.0)

def test_from_dataset_infers_keys_and_freq():
    ts = _make_touchstone()
    freq = ts.network.f

    codec = TouchstoneCodec.from_dataset(_DummyTouchstoneDataset())
    assert codec.freq_hz.shape == freq.shape
    assert set(codec.x_keys) == {"a", "b"}
    assert codec.n_ports == 2

def test_backup_mode_full(ts_sample, freq_vec):
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real"],  # только Re(S11)
        freq_hz=freq_vec,
        backup_mode="full",
    )
    _, y_t, meta = codec.encode(ts_sample)
    assert "s_backup" in meta

    ts_rec = codec.decode(y_t, meta)
    s_ref = ts_sample.network.s
    s_rec = ts_rec.network.s

    # Только Re(S11) предсказывался
    np.testing.assert_allclose(s_rec[:, 0, 0].real, s_ref[:, 0, 0].real, atol=1e-6)
    # Остальные элементы — из s_backup
    np.testing.assert_allclose(s_rec[:, 0, 1], s_ref[:, 0, 1], atol=1e-6)
    np.testing.assert_allclose(s_rec[:, 1, 0], s_ref[:, 1, 0], atol=1e-6)
    np.testing.assert_allclose(s_rec[:, 1, 1], s_ref[:, 1, 1], atol=1e-6)

def test_backup_mode_missing(ts_sample, freq_vec):
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real"],  # только Re(S11)
        freq_hz=freq_vec,
        backup_mode="missing",
    )
    _, y_t, meta = codec.encode(ts_sample)

    # Проверяем, что резерв действительно сохранён выборочно
    assert "s_missing" in meta
    assert "s_backup" not in meta
    assert "0_0" not in meta["s_missing"]  # т.к. S1_1 предсказывается
    assert "0_1" in meta["s_missing"]
    assert "1_1" in meta["s_missing"]

    ts_rec = codec.decode(y_t, meta)
    s_ref = ts_sample.network.s
    s_rec = ts_rec.network.s

    # Проверяем предсказанный канал
    np.testing.assert_allclose(s_rec[:, 0, 0].real, s_ref[:, 0, 0].real, atol=1e-6)
    # Проверяем восстановленные по резерву каналы
    np.testing.assert_allclose(s_rec[:, 0, 1], s_ref[:, 0, 1], atol=1e-6)
    np.testing.assert_allclose(s_rec[:, 1, 0], s_ref[:, 1, 0], atol=1e-6)
    np.testing.assert_allclose(s_rec[:, 1, 1], s_ref[:, 1, 1], atol=1e-6)

def test_backup_mode_none(ts_sample, freq_vec):
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real"],
        freq_hz=freq_vec,
        backup_mode="none",
    )
    _, y_t, meta = codec.encode(ts_sample)
    assert "s_backup" not in meta
    assert "s_missing" not in meta

    ts_rec = codec.decode(y_t, meta)
    s_rec = ts_rec.network.s
    # Проверяем, что отсутствующие элементы заполнены значениями по умолчанию
    assert np.allclose(s_rec[:, 0, 0].real, ts_sample.network.s[:, 0, 0].real, atol=1e-6)
    assert np.allclose(s_rec[:, 0, 1], 0.0, atol=1e-6)

def test_encode_decode_s(ts_sample, freq_vec):
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real", "S1_1.imag", "S1_1.gd"],
        freq_hz=freq_vec,
    )
    y_t, meta = codec.encode_s(ts_sample.network)
    ts_rec = codec.decode_s(y_t, meta)
    np.testing.assert_allclose(ts_rec.s[:, 0, 0],
                               ts_sample.network.s[:, 0, 0],
                               atol=1e-6)

def test_encode_decode_x(ts_sample, freq_vec):
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S1_1.real", "S1_1.imag"],
        freq_hz=freq_vec,
    )
    x_t = codec.encode_x(ts_sample.params)
    params_rec = codec.decode_x(x_t)
    assert params_rec["a"] == ts_sample.params["a"]
    assert params_rec["b"] == ts_sample.params["b"]


# ---------------------------------------------------------------- pickle
def test_pickle_roundtrip(ts_sample, freq_vec):
    y_ch = ["S1_1.real", "S1_1.imag"]
    codec = TouchstoneCodec(
        x_keys=["a"],
        y_channels=y_ch,
        freq_hz=freq_vec,
        backup_mode="full",  # ← важно для полной реконструкции
    )

    data = codec.dumps()
    codec2 = TouchstoneCodec.loads(data)

    x_t, y_t, meta = codec2.encode(ts_sample)
    ts_rec = codec2.decode(y_t, meta)

    np.testing.assert_allclose(ts_rec.network.s, ts_sample.network.s, atol=1e-6)

def test_pickle_roundtrip_partial(ts_sample, freq_vec):
    y_ch = ["S1_1.real", "S1_1.imag"]
    codec = TouchstoneCodec(
        x_keys=["a"],
        y_channels=y_ch,
        freq_hz=freq_vec,
        backup_mode="none",  # ← тестируем поведение без резерва
    )

    data = codec.dumps()
    codec2 = TouchstoneCodec.loads(data)

    x_t, y_t, meta = codec2.encode(ts_sample)
    ts_rec = codec2.decode(y_t, meta)

    s_rec = ts_rec.network.s
    s_ref = ts_sample.network.s

    # Сравниваем только реально предсказанные компоненты
    np.testing.assert_allclose(s_rec[:, 0, 0].real, s_ref[:, 0, 0].real, atol=1e-6)
    np.testing.assert_allclose(s_rec[:, 0, 0].imag, s_ref[:, 0, 0].imag, atol=1e-6)
