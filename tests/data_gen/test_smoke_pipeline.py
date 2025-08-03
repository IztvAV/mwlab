"""tests/data_gen/test_smoke_pipeline.py
================================================
Smoke-тест full pipeline с CMGenerator и разными Writer-ами:
* TouchstoneDirWriter
* HDF5Writer

Все тяжёлые зависимости подменяются заглушками. Проверяем, что:
- TouchstoneDirWriter создаёт ровно N файлов .sNp;
- HDF5Writer создаёт файл и количество append == N.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1) Заглушки зависимостей + «пустой» корневой пакет mwlab
# ─────────────────────────────────────────────────────────────────────────────

def _install_fakes(monkeypatch: pytest.MonkeyPatch):
    """Создаёт минимальные фейки для зависимостей и регистрирует их в sys.modules."""

    def _get_or_create(name: str) -> ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = ModuleType(name)
            monkeypatch.setitem(sys.modules, name, mod)
        return mod

    # --- Корневой пакет mwlab (не исполняем реальный __init__.py) ---
    root = _get_or_create("mwlab")
    if not hasattr(root, "__path__"):
        repo_root = Path(__file__).resolve().parents[2]
        pkg_dir = repo_root / "mwlab"
        root.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]

    # --- torch (минимум) ---
    torch_mod = _get_or_create("torch")
    if not hasattr(torch_mod, "__path__"):
        torch_mod.__path__ = []  # type: ignore[attr-defined]

    class _Tensor:
        def __init__(self, arr): self._a = np.asarray(arr)
        def to(self, *_, **__): return self
        def flatten(self): return _Tensor(self._a.ravel())
        def numel(self): return int(self._a.size)
        def cpu(self): return self
        def numpy(self): return self._a
        def detach(self): return self
        @property
        def shape(self): return self._a.shape
        def __getitem__(self, idx): return _Tensor(self._a[idx])

    def as_tensor(x, dtype=None, device=None):  # noqa: ARG001
        return _Tensor(x)

    class _Device:
        def __init__(self, name): self.type = str(name)
        def __repr__(self): return f"device({self.type})"

    class _NoGrad:
        def __call__(self, f=None):
            # как декоратор без аргументов
            return self if f is None else f
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    def stack(tensors: Sequence[_Tensor], dim: int = 0):  # noqa: ARG001
        arrs = [t._a if isinstance(t, _Tensor) else np.asarray(t) for t in tensors]
        return _Tensor(np.stack(arrs, axis=0))

    setattr(torch_mod, "Tensor", _Tensor)
    setattr(torch_mod, "as_tensor", as_tensor)
    setattr(torch_mod, "device", _Device)
    setattr(torch_mod, "no_grad", _NoGrad())
    setattr(torch_mod, "stack", stack)

    # torch.utils.data.Dataset (на всякий случай)
    utils_mod = _get_or_create("torch.utils")
    if not hasattr(utils_mod, "__path__"):
        utils_mod.__path__ = []  # type: ignore[attr-defined]
    data_mod = _get_or_create("torch.utils.data")
    class _Dataset: ...
    setattr(data_mod, "Dataset", _Dataset)

    # --- skrf (минимум) ---
    skrf_mod = _get_or_create("skrf")
    class _Frequency:
        @classmethod
        def from_f(cls, f: np.ndarray, unit="Hz"):  # noqa: ARG002
            obj = cls(); obj.f = np.asarray(f, dtype=float); obj.unit = unit; return obj
    class _Network:
        def __init__(self, frequency, s):
            self.frequency = frequency
            self.s = np.asarray(s)
            self.number_of_ports = int(self.s.shape[-1])
            self.comments = ""
            self.s_def = "S"
            self.z0 = np.array([50.0])
        def copy(self): return _Network(self.frequency, self.s.copy())
        @property
        def f(self): return self.frequency.f
        def resample(self, freq_target):  # pragma: no cover
            self.frequency = freq_target
    setattr(skrf_mod, "Frequency", _Frequency)
    setattr(skrf_mod, "Network", _Network)

    # --- mwlab.filters.cm_core ---
    cm_core = _get_or_create("mwlab.filters.cm_core")
    class _CoreSpec:
        def __init__(self, order: int, ports: int, method: str):  # noqa: ARG002
            self.order = order; self.ports = ports
    def solve_sparams(spec: _CoreSpec, M_real, omega, **kwargs):  # noqa: ARG001
        def _arr(x): return getattr(x, "_a", np.asarray(x))
        F = int(np.size(_arr(omega))); P = int(spec.ports)
        M = _arr(M_real); B = int(M.shape[0]) if getattr(M, "ndim", 0) else 1
        out = np.zeros((B, F, P, P), dtype=np.complex128)
        for b in range(B):
            for f in range(F):
                for p in range(P):
                    out[b, f, p, p] = 1.0 + 0.0j
        return _Tensor(out)
    setattr(cm_core, "CoreSpec", _CoreSpec)
    setattr(cm_core, "solve_sparams", solve_sparams)
    setattr(cm_core, "DEFAULT_DEVICE", "cpu")
    setattr(cm_core, "DT_R", None)

    # --- mwlab.filters.cm_schema ---
    cm_schema = _get_or_create("mwlab.filters.cm_schema")
    class _AllowAll:
        def __contains__(self, k): return True
    class _ParamSchema:
        def __init__(self, expected_keys: Sequence[str] | None = None):
            self._keys = set(expected_keys) if expected_keys is not None else _AllowAll()
        @property
        def keys(self): return self._keys
        @classmethod
        def from_topology(cls, topo): return cls(None)
        def pack(self, p: Mapping[str, Any], *, device=None, dtype=None, strict=True):  # noqa: ARG002
            keys = [k for k in sorted(p.keys()) if not k.startswith("__")]
            vec = np.array([float((hash(k) % 7) + 1) for k in keys], dtype=float)
            return as_tensor(vec)
        def assemble(self, vecs, *, device=None):  # noqa: ARG002
            arr = getattr(vecs, "_a", np.asarray(vecs)); return as_tensor(arr), None, None, None
    setattr(cm_schema, "ParamSchema", _ParamSchema)

    # --- mwlab.filters.topologies ---
    tops = _get_or_create("mwlab.filters.topologies")
    class _Topology:
        def __init__(self, order: int, ports: int, name: str = ""):
            self.order = order; self.ports = ports; self.name = name
    setattr(tops, "Topology", _Topology)

    # --- mwlab.io.touchstone ---
    touch = _get_or_create("mwlab.io.touchstone")
    class _TouchstoneData:
        def __init__(self, network, params: Mapping[str, Any] | None = None):
            self.network = network; self.params = dict(params or {}); self.path = None
        def save(self, path: str | Path):
            path = Path(path); path.write_text("# fake touchstone\n")
            self.path = str(path)
    setattr(touch, "TouchstoneData", _TouchstoneData)

    # --- mwlab.io.backends.hdf5_backend (HDF5Backend) ---
    bmod = _get_or_create("mwlab.io.backends.hdf5_backend")
    class _H5Obj:
        def __init__(self, backend): self._b = backend
        def flush(self):  # пишем JSON-сводку в файл
            self._b._write_summary()
    class HDF5Backend:
        def __init__(self, path: str | Path, mode: str = "a", in_memory: bool = False):  # noqa: ARG002
            self.path = Path(path); self.count = 0; self.h5 = _H5Obj(self)
            # создаём пустой файл сразу
            self.path.write_text(json.dumps({"count": 0}))
        def append(self, ts_obj):  # noqa: ARG001
            self.count += 1
        def _write_summary(self):
            self.path.write_text(json.dumps({"count": self.count}))
        def close(self):
            self._write_summary()
    setattr(bmod, "HDF5Backend", HDF5Backend)

    # --- mwlab.filters.devices (для DeviceCMGenerator, если понадобится) ---
    devm = _get_or_create("mwlab.filters.devices")
    class _DevCM:
        def __init__(self, topo): self.topo = topo
    class _Device:
        def __init__(self, topo, scale: float = 1.0, extras: Mapping[str, Any] | None = None):
            self.cm = _DevCM(topo); self._scale = float(scale); self._extras = dict(extras or {})
        def _omega(self, f_arr: np.ndarray): return np.asarray(f_arr, dtype=float) / max(self._scale, 1.0)
        def _device_params(self): return {"device_tag": "stub", **self._extras}
    setattr(devm, "Device", _Device)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Фикстура окружения smoke-pipeline
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def smoke_env(monkeypatch: pytest.MonkeyPatch):
    _install_fakes(monkeypatch)
    import mwlab.data_gen.cm_generators as cg
    return {"cg": cg}


# ─────────────────────────────────────────────────────────────────────────────
# 3) Тесты
# ─────────────────────────────────────────────────────────────────────────────

def test_touchstone_dir_writer_creates_snp_files(smoke_env, tmp_path: Path):
    cg = smoke_env["cg"]
    topo = cg.Topology(order=3, ports=2, name="2port")
    omega = np.linspace(-1.0, 1.0, 9)
    gen = cg.CMGenerator(topology=topo, omega_grid=omega)

    # Источник/приёмник/раннер (реальные модули из проекта)
    from mwlab.data_gen.sources import ListSource
    from mwlab.data_gen.writers import TouchstoneDirWriter
    from mwlab.data_gen.runner import run_pipeline

    N = 5
    pts = [{"v": i} for i in range(N)]
    src = ListSource(pts, shuffle=False, copy=True)

    out_dir = tmp_path / "snp_out"
    wr = TouchstoneDirWriter(out_dir, stem_format="smp_{idx:03d}", overwrite=True)

    run_pipeline(src, gen, wr, batch_size=2, progress=False)

    # Должно быть N файлов .s2p
    files = sorted(p for p in out_dir.iterdir() if p.suffix.lower().endswith("p"))
    assert len(files) == N
    # Имена соответствуют шаблону
    assert files[0].name.startswith("smp_000")
    assert files[-1].name.endswith(".s2p")


def test_hdf5_writer_creates_file_and_appends_count(smoke_env, tmp_path: Path):
    cg = smoke_env["cg"]
    topo = cg.Topology(order=4, ports=3, name="3port")
    omega = np.linspace(-2.0, 2.0, 11)
    gen = cg.CMGenerator(topology=topo, omega_grid=omega)

    from mwlab.data_gen.sources import ListSource
    from mwlab.data_gen.writers import HDF5Writer
    from mwlab.data_gen.runner import run_pipeline

    N = 7
    pts = [{"k": i} for i in range(N)]
    src = ListSource(pts, shuffle=False, copy=True)

    h5_path = tmp_path / "out.h5"
    wr = HDF5Writer(h5_path, compression=None, overwrite=True)

    run_pipeline(src, gen, wr, batch_size=3, progress=False)

    # Файл должен существовать, а внутри JSON-сводка с count==N (наша фейковая реализация)
    assert h5_path.exists()
    info = json.loads(h5_path.read_text())
    assert info.get("count") == N
