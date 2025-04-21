# transforms/__init__.py:
class TComposite:
    """
    Последовательная композиция трансформаций.

    Применяет все трансформы из списка по порядку:
        x = tf(x)
    """
    def __init__(self, tfs):
        self.tfs = tfs

    def __call__(self, x):
        for tf in self.tfs:
            x = tf(x)
        return x
