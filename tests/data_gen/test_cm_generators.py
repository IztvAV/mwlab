"""tests/data_gen/test_cm_generators.py
================================================
Юнит-тесты для mwlab.data_gen.cm_generators без тяжёлых зависимостей.

Мы подменяем sys.modules заглушками для:
- torch (как пакет с Tensor/as_tensor/device/no_grad/stack, utils.data.Dataset),
- skrf (Frequency/Network),
- mwlab.filters.cm_core (solve_sparams/CoreSpec/DEFAULT_DEVICE/DT_R),
- mwlab.filters.cm_schema (ParamSchema),
- mwlab.filters.topologies (Topology),
- mwlab.io.touchstone (TouchstoneData),
- mwlab.filters.devices (Device).

КЛЮЧЕВОЕ: создаём «пустой» корневой пакет `mwlab` (без выполнения его __init__.py),
но с корректным __path__, указывающим на реальную директорию проекта `mwlab/`.
Так импорт `mwlab.data_gen.cm_generators` пойдёт с диска, минуя тяжёлые зависимые
пакеты из `mwlab/__init__.py`.
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1) Заглушки внешних модулей + подмена корня `mwlab`
# ---------------------------------------------------------------------------

def _install_fakes(monkeypatch: pytest.MonkeyPatch):
    """Создаёт и регистрирует в sys.modules компактные фейки зависимостей."""

    # ---------- helper ----------
    def _get_or_create(name: str) -> ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = ModuleType(name)
            monkeypatch.setitem(sys.modules, name, mod)
        return mod

    # ---------- СНАЧАЛА: подменяем корневой пакет `mwlab` ----------
    # Делаем "пустой" пакет, но с __path__ → реальная директория проекта /mwlab
    mwlab_root = _get_or_create("mwlab")
    if not hasattr(mwlab_root, "__path__"):
        # tests/…/this_file.py → …/.. (repo root) → /mwlab
        repo_root = Path(__file__).resolve().parents[2]
        pkg_dir = repo_root / "mwlab"
        mwlab_root.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    # ВАЖНО: не импортируем реальный mwlab.__init__.py!

    # ---------- fake torch ----------
    torch_mod = _get_or_create("torch")
    # сделать «пакетом»
    if not hasattr(torch_mod, "__path__"):
        torch_mod.__path__ = []  # type: ignore[attr-defined]

    class _FakeTensor:
        def __init__(self, arr):
            self._a = np.asarray(arr)

        def to(self, *_, **__):
            return self

        def flatten(self):
            return _FakeTensor(self._a.ravel())

        def numel(self):
            return int(self._a.size)

        def cpu(self):
            return self

        def numpy(self):
            return self._a

        def detach(self):
            return self

        @property
        def shape(self):
            return self._a.shape

        def __getitem__(self, idx):
            return _FakeTensor(self._a[idx])

    def as_tensor(x, dtype=None, device=None):  # noqa: ARG001
        return _FakeTensor(x)

    class _FakeDevice:
        def __init__(self, name):
            self.type = str(name)

        def __repr__(self):
            return f"device({self.type})"

    class _NoGrad:
        """Имитация контекст-менеджера + вызываемого декоратора: @torch.no_grad()."""
        def __call__(self, *args, **kwargs):  # как декоратор без аргументов
            return self
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    def stack(tensors: Sequence[_FakeTensor], dim: int = 0):  # noqa: ARG001
        arrs = [t._a if isinstance(t, _FakeTensor) else np.asarray(t) for t in tensors]
        return _FakeTensor(np.stack(arrs, axis=0))

    setattr(torch_mod, "Tensor", _FakeTensor)
    setattr(torch_mod, "as_tensor", as_tensor)
    setattr(torch_mod, "device", _FakeDevice)
    setattr(torch_mod, "no_grad", _NoGrad())  # декоратор/контекст – ИНСТАНС
    setattr(torch_mod, "stack", stack)

    # torch.utils / torch.utils.data.Dataset
    utils_mod = _get_or_create("torch.utils")
    if not hasattr(utils_mod, "__path__"):
        utils_mod.__path__ = []  # type: ignore[attr-defined]
    data_mod = _get_or_create("torch.utils.data")

    class _Dataset:  # минимальная заглушка
        pass

    setattr(data_mod, "Dataset", _Dataset)

    # ---------- fake skrf ----------
    skrf_mod = _get_or_create("skrf")

    class _Frequency:
        @classmethod
        def from_f(cls, f: np.ndarray, unit="Hz"):  # noqa: ARG002
            obj = cls()
            obj.f = np.asarray(f, dtype=float)
            obj.unit = unit
            return obj

    class _Network:
        def __init__(self, frequency, s):
            self.frequency = frequency
            self.s = np.asarray(s)
            self.number_of_ports = int(self.s.shape[-1])

    setattr(skrf_mod, "Frequency", _Frequency)
    setattr(skrf_mod, "Network", _Network)

    # ---------- fake mwlab.filters.cm_core ----------
    cm_core_mod = _get_or_create("mwlab.filters.cm_core")

    class _CoreSpec:
        def __init__(self, order: int, ports: int, method: str):  # noqa: ARG002
            self.order = order
            self.ports = ports

    def solve_sparams(spec: _CoreSpec, M_real, omega, **kwargs):  # noqa: ARG001
        """Вернуть «тензор» формы (B,F,P,P) с единичной диагональю."""
        def _to_arr(x):
            return getattr(x, "_a", np.asarray(x))

        omega_arr = _to_arr(omega)
        F = int(np.size(omega_arr))
        M = _to_arr(M_real)
        B = int(M.shape[0]) if getattr(M, "ndim", 0) >= 1 else 1
        P = int(spec.ports)
        out = np.zeros((B, F, P, P), dtype=np.complex128)
        for b in range(B):
            for f in range(F):
                for p in range(P):
                    out[b, f, p, p] = 1.0 + 0.0j
        return _FakeTensor(out)

    setattr(cm_core_mod, "solve_sparams", solve_sparams)
    setattr(cm_core_mod, "CoreSpec", _CoreSpec)
    setattr(cm_core_mod, "DEFAULT_DEVICE", "cpu")
    setattr(cm_core_mod, "DT_R", None)

    # ---------- fake mwlab.filters.cm_schema ----------
    cm_schema_mod = _get_or_create("mwlab.filters.cm_schema")

    class _AllowAllKeys:
        def __contains__(self, item):
            return True

    class _ParamSchema:
        """Мини-схема: pack/assemble и from_topology(), совместимая с CMGenerator."""

        def __init__(self, expected_keys: Sequence[str] | None = None):
            self._keys = set(expected_keys) if expected_keys is not None else _AllowAllKeys()

        @property
        def keys(self):
            return self._keys

        @classmethod
        def from_topology(cls, topo):
            return cls(expected_keys=None)

        def pack(self, p: Mapping[str, Any], *, device=None, dtype=None, strict=True):  # noqa: ARG002
            keys = [k for k in sorted(p.keys()) if not k.startswith("__")]
            vec = np.array([float((hash(k) % 7) + 1) for k in keys], dtype=float)
            return as_tensor(vec)

        def assemble(self, vecs, *, device=None):  # noqa: ARG002
            arr = getattr(vecs, "_a", np.asarray(vecs))
            return as_tensor(arr), None, None, None  # (M_real, qu, phase_a, phase_b)

    setattr(cm_schema_mod, "ParamSchema", _ParamSchema)

    # ---------- fake mwlab.filters.topologies ----------
    topologies_mod = _get_or_create("mwlab.filters.topologies")

    class _Topology:
        def __init__(self, order: int, ports: int, name: str = ""):
            self.order = order
            self.ports = ports
            self.name = name

    setattr(topologies_mod, "Topology", _Topology)

    # ---------- fake mwlab.io.touchstone ----------
    touch_mod = _get_or_create("mwlab.io.touchstone")

    class _TouchstoneData:
        def __init__(self, network, params: Mapping[str, Any] | None = None):
            self.network = network
            self.params = dict(params or {})
            self.path = None  # для совместимости с некоторыми кодеками

        def save(self, *_, **__):  # pragma: no cover
            pass

    setattr(touch_mod, "TouchstoneData", _TouchstoneData)

    # ---------- fake mwlab.filters.devices ----------
    devices_mod = _get_or_create("mwlab.filters.devices")

    class _DevCM:
        def __init__(self, topo):
            self.topo = topo

    class _Device:
        def __init__(self, topo, scale: float = 1.0, extras: Mapping[str, Any] | None = None):
            self.cm = _DevCM(topo)
            self._scale = float(scale)
            self._extras = dict(extras or {})

        def _omega(self, f_arr: np.ndarray):
            return np.asarray(f_arr, dtype=float) / max(self._scale, 1.0)

        def _device_params(self):
            return {"device_tag": "stub", **self._extras}

    setattr(devices_mod, "Device", _Device)

    # ВАЖНО: больше не чистим sys.modules['mwlab*'], чтобы не стереть наш корневой пакет.


# ---------------------------------------------------------------------------
# 2) Фикстура окружения: ставим фейки и импортируем тестируемый модуль
# ---------------------------------------------------------------------------

@pytest.fixture()
def cm_env(monkeypatch: pytest.MonkeyPatch):
    _install_fakes(monkeypatch)
    import mwlab.data_gen.cm_generators as cg  # импорт после подмен
    return {"cg": cg}


# ---------------------------------------------------------------------------
# 3) Тесты
# ---------------------------------------------------------------------------

def test_cm_generator_smoke_pipeline(cm_env):
    cg = cm_env["cg"]
    topo = cg.Topology(order=3, ports=2, name="T2")
    omega = np.linspace(-1.0, 1.0, 5)

    gen = cg.CMGenerator(topology=topo, omega_grid=omega)
    pts = [{"a": 1.0}, {"b": 2.0}]

    from mwlab.data_gen.sources import ListSource
    from mwlab.data_gen.writers import ListWriter
    from mwlab.data_gen.runner import run_pipeline

    src = ListSource(pts, shuffle=False, copy=True)
    wr = ListWriter()
    run_pipeline(src, gen, wr, batch_size=2, progress=False)

    result = wr.result()
    assert len(result["data"]) == 2
    td0 = result["data"][0]
    s_shape = td0.network.s.shape
    assert s_shape == (len(omega), topo.ports, topo.ports)
    m0 = result["meta"][0]
    assert m0["topo_order"] == topo.order
    assert m0["topo_ports"] == topo.ports
    assert m0["topo_name"] == topo.name
    assert m0["freq_normalized"] is True
    assert m0["grid_size"] == len(omega)


def test_device_cm_generator_adds_device_meta_and_real_freq(cm_env):
    cg = cm_env["cg"]
    topo = cg.Topology(order=4, ports=3, name="Tri")

    from mwlab.filters.devices import Device  # fake
    dev = Device(topo, scale=1e9, extras={"hw_rev": "A1"})

    f_grid = np.array([1.0e9, 1.5e9, 2.0e9])

    gen = cg.DeviceCMGenerator(device=dev, f_grid=f_grid)
    pts = [{"p": 1.0}, {"p": 2.0}, {"p": 3.0}]

    from mwlab.data_gen.sources import ListSource
    from mwlab.data_gen.writers import ListWriter
    from mwlab.data_gen.runner import run_pipeline

    src = ListSource(pts, shuffle=False, copy=True)
    wr = ListWriter()
    run_pipeline(src, gen, wr, batch_size=2, progress=False)

    res = wr.result()
    assert len(res["data"]) == 3
    for m in res["meta"]:
        assert m["freq_normalized"] is False
        assert m["hw_rev"] == "A1"
        assert m["topo_ports"] == topo.ports


def test_preprocess_rejects_unknown_keys_when_schema_is_strict(cm_env):
    """Если схема параметров задаёт ожидаемые ключи — неизвестные ключи отвергаются."""
    cg = cm_env["cg"]
    from mwlab.filters.cm_schema import ParamSchema  # fake

    topo = cg.Topology(order=2, ports=2, name="pair")
    strict_schema = ParamSchema(expected_keys=["x", "y"])

    gen = cg.CMGenerator(topology=topo, omega_grid=[0.0, 1.0], schema=strict_schema)

    with pytest.raises(KeyError):
        gen.preprocess({"x": 1, "z": 2})
