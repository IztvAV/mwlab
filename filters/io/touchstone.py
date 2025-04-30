from mwlab import TouchstoneData
import pathlib
from typing import Union, Any, Dict
from filters.filter.mwfilter import MWFilter


class TouchstoneMWFilterData(TouchstoneData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _params_from_file(path: pathlib.Path) -> Dict[str, Any]:
        net = MWFilter(str(path))
        return {"f0": net.f0, "bw": net.bw, "Q": net.Q, "matrix": net.coupling_matrix.factors}

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "TouchstoneData":
        path = pathlib.Path(path)
        net = MWFilter(str(path))
        obj = cls(net, path=path)

        # запасной парсинг, если comments пуст
        if not obj.params:
            obj.params = obj._params_from_file(path)

        return obj