# datasets/touchstone_dataset.py
import torch, numpy as np, pathlib
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional
from mwlab import TouchstoneData

class TouchstoneDataset(Dataset):
    """Итератор по *.sNp‑файлам в директории.

    *При создании* мы только собираем список путей.
    *При __getitem__* читаем нужный файл -> TouchstoneFile.from_path()
    и применяем трансформы.
    """

    def __init__(self,
                 root: str | pathlib.Path,
                 pattern: str = "*.s?p",
                 x_keys: Optional[Sequence[str]] = None,
                 x_tf: Optional[Callable] = None,
                 s_tf: Optional[Callable] = None):
        self.root    = pathlib.Path(root)
        self.paths   = sorted(self.root.rglob(pattern))
        self.x_keys  = x_keys        # если None – берём все ключи
        self.x_tf    = x_tf
        self.s_tf    = s_tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ts = TouchstoneData.load(self.paths[idx])  # 💡

        # -------- X (параметры модели) ----------
        x = {k: ts.params.get(k, np.nan) for k in self.x_keys or ts.params}
        if self.x_tf:
            x = self.x_tf(x)

        # -------- S‑параметры --------------------
        net = ts.network  # S‑данные
        s = self.s_tf(net) if self.s_tf else net

        return x, s