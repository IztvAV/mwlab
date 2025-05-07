from mwlab import TouchstoneData
import pathlib
from typing import Union, Any, Dict
from filters.filter.mwfilter import MWFilter


class TouchstoneMWFilterData(TouchstoneData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "TouchstoneData":
        path = pathlib.Path(path)
        net = MWFilter.from_file(str(path))
        obj = cls(net, path=path)

        # запасной парсинг, если comments пуст
        if not obj.params:
            obj.params = obj._params_from_file(path)

        return obj