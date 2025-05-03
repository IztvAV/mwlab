from mwlab import TouchstoneDataset
import mwlab.datasets.touchstone_dataset as td
from mwlab.io.backends import StorageBackend
from filters.io.touchstone import TouchstoneMWFilterData
from filters.io.backends import MWFilterFileBackend
import numpy as np
import pathlib


def get_backend(path: pathlib.Path,
                pattern: str = "*.s?p") -> StorageBackend:
    """
    Возвращает подходящий backend по типу *path*.
    • Каталог           → FileBackend(root)
    • .h5 / .hdf5       → HDF5Backend(path, 'r')
    • .zarr             → ZarrBackend(path, 'r')   # когда появится
    • иначе ValueError
    """
    if path.is_dir():
        return MWFilterFileBackend(path, pattern)
    else:
        return td.get_backend(path, pattern)


class TouchstoneMWFilterDataset(TouchstoneDataset):
    """Итератор по *.sNp‑файлам в директории.

    *При создании* мы только собираем список путей.
    *При __getitem__* читаем нужный файл -> TouchstoneFile.from_path()
    и применяем трансформы.
    """

    def __init__(
            self,
            source: StorageBackend | str | pathlib.Path,
            *,
            x_keys=None,
            x_tf=None,
            s_tf=None,
            pattern="*.s?p"
    ):
        super().__init__(source, x_keys=x_keys, x_tf=x_tf, s_tf=s_tf, pattern=pattern)
        if isinstance(source, (str, pathlib.Path)):
            backend = get_backend(pathlib.Path(source), pattern)
        else:
            backend = source   # уже готовый backend

        self.backend = backend

    def __getitem__(self, idx):
        ts = self.backend.read(idx)  # TouchstoneData

        # ----------- X (скалярные параметры) -----------
        x = {k: ts.params.get(k, np.nan) for k in (self.x_keys or ts.params)}
        if self.x_tf:
            x = self.x_tf(x)

        # ----------- S-матрица --------------------------
        net = ts.network
        s_out = self.s_tf(net) if self.s_tf else net

        return x, s_out

