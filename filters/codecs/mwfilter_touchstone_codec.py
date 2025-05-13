from mwlab import TouchstoneCodec


class MWFilterTouchstoneCodec(TouchstoneCodec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def exclude_keys(self, keys_to_exclude=["f0", "bw", "N", "Q"]):  # Q исключаю сейчас для простоты
        # Исключаем из анализа ненужные x-параметры
        self.x_keys = list(filter(lambda x: x not in keys_to_exclude, self.x_keys))
