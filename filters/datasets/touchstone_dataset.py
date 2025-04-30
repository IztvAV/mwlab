from mwlab import TouchstoneDataset
from filters.io.touchstone import TouchstoneMWFilterData
import numpy as np


class TouchstoneMWFilterDataset(TouchstoneDataset):
    """Итератор по *.sNp‑файлам в директории.

    *При создании* мы только собираем список путей.
    *При __getitem__* читаем нужный файл -> TouchstoneFile.from_path()
    и применяем трансформы.
    """

    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        ts = TouchstoneMWFilterData.load(self.paths[idx])  # 💡

        # -------- X (параметры модели) ----------
        x = {k: ts.params.get(k, np.nan) for k in self.x_keys or ts.params}
        if self.x_tf:
            x = self.x_tf(x)

        # -------- S‑параметры --------------------
        net = ts.network  # S‑данные
        s = self.s_tf(net) if self.s_tf else net

        return x, s
