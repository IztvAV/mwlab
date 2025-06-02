from mwlab import TouchstoneCodec


class MWFilterTouchstoneCodec(TouchstoneCodec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ----------------------------------------------------------------- factory
    @classmethod
    def from_dataset(cls, keys_for_analysis, *args, **kwargs) -> "MWFilterTouchstoneCodec":
        obj = super().from_dataset(*args, **kwargs)
        # Удаляем ненужные индексы
        obj.x_keys = keys_for_analysis
        return obj

    def exclude_keys(self, keys_to_exclude=["f0", "bw", "N", "Q"]):  # Q исключаю сейчас для простоты
        # Исключаем из анализа ненужные x-параметры
        self.x_keys = list(filter(lambda x: x not in keys_to_exclude, self.x_keys))
