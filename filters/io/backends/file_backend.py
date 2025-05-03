from mwlab.io.backends import FileBackend
from filters.io.touchstone import TouchstoneMWFilterData


class MWFilterFileBackend(FileBackend):
    """Backend, который хранит набор путей к *.sNp."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self, idx: int) -> TouchstoneMWFilterData:  # noqa: D401
        return TouchstoneMWFilterData.load(self.paths[idx])
