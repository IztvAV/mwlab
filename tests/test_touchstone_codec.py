# tests/test_touchstone_codec.py
"""
Набор базовых проверок для TouchstoneCodec.

Покрывает:
    ✓ round-trip  encode → decode  (real/imag и db/deg);
    ✓ корректную обработку неполных каналов (amp без фазы);
    ✓ валидацию формата каналов (_parse_channel).
"""
import numpy as np
import pytest
import skrf as rf

from mwlab.codecs.touchstone_codec import TouchstoneCodec
from mwlab.io.touchstone import TouchstoneData


# ─────────────────────────────────────────────────────────────────────────────
#                                       HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_touchstone(n_ports: int = 2, n_freq: int = 101) -> TouchstoneData:
    """Генерируем стабильные искусственные данные."""
    rng = np.random.default_rng(seed=0)
    f = np.linspace(1e9, 5e9, n_freq)  # 1–5 ГГц
    s = (rng.standard_normal((n_freq, n_ports, n_ports)) +
         1j * rng.standard_normal((n_freq, n_ports, n_ports))) * 0.1
    net = rf.Network(f=f, s=s, f_unit="Hz")
    params = {"a": 1.0, "b": 2.0}
    return TouchstoneData(net, params=params)


def _assert_network_equal(a: rf.Network, b: rf.Network, *, atol=1e-8, rtol=1e-6):
    """Проверяем эквивалентность сетей."""
    assert a.number_of_ports == b.number_of_ports, "Number of ports mismatch"
    assert np.allclose(a.f, b.f, atol=0, rtol=1e-12), "Frequency vectors mismatch"
    assert np.allclose(a.s, b.s, atol=atol, rtol=rtol), "S-parameters mismatch"


# ─────────────────────────────────────────────────────────────────────────────
#                                       FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ts_sample():
    """Пример искусственной TouchstoneData."""
    return _make_touchstone()


@pytest.fixture(scope="module")
def freq_vec(ts_sample):
    """Вектор частот из TouchstoneData."""
    return ts_sample.network.f


# ─────────────────────────────────────────────────────────────────────────────
#                                       TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_roundtrip_real_imag(ts_sample, freq_vec):
    """
    encode+decode с каналами real/imag восстанавливает исходную сеть.
    """
    y_ch = [f"S{i}{j}.{p}" for p in ("real", "imag") for i in (1, 2) for j in (1, 2)]
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=y_ch,
        freq_hz=freq_vec,
    )

    x_t, y_t, meta = codec.encode(ts_sample)
    ts_rec = codec.decode(y_t, meta)

    _assert_network_equal(ts_sample.network, ts_rec.network)


def test_roundtrip_db_deg(ts_sample, freq_vec):
    """
    encode+decode с каналами db/deg восстанавливает исходную сеть.
    (больше допуски по atol из-за фазовых преобразований)
    """
    y_ch = [f"S{i}{j}.{p}" for p in ("db", "deg") for i in (1, 2) for j in (1, 2)]
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=y_ch,
        freq_hz=freq_vec,
    )

    x_t, y_t, meta = codec.encode(ts_sample)
    ts_rec = codec.decode(y_t, meta)

    _assert_network_equal(ts_sample.network, ts_rec.network, atol=1e-6)


def test_amp_only_nan_fill(ts_sample, freq_vec):
    """
    Если переданы только amp-каналы → Phase = 0.
    Проверяем:
      • модуль совпадает;
      • мнимая часть ≈ 0.
    """
    y_ch = [f"S{i}{j}.mag" for i in (1, 2) for j in (1, 2)]
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

    # модуль совпадает
    assert np.allclose(np.abs(s_rec), np.abs(s_ref), rtol=1e-6, atol=1e-9), "Modulus mismatch"
    # фаза нулевая → мнимая часть ≈ 0
    assert np.allclose(s_rec.imag, 0.0, atol=1e-9), "Imaginary part not zero"


@pytest.mark.parametrize("bad_tag", ["S1X.real", "S223.imag", "S11.foo"])
def test_parse_channel_validation(bad_tag):
    """
    Неверный формат каналов приводит к ValueError.
    """
    # создаём нормальный codec, а проверяем только _parse_channel
    codec = TouchstoneCodec(
        x_keys=["a", "b"],
        y_channels=["S11.real"],  # корректный канал
        freq_hz=np.array([1.0]),
    )
    with pytest.raises(ValueError):
        codec._parse_channel(bad_tag)

