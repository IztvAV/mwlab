# transforms/x_transforms.py
class X_SelectKeys:
    """
    Оставляет в словаре только указанные ключи.

    Пример:
        x = {"w": 1.2, "l": 0.9, "gap": 0.1}
        X_SelectKeys(["w", "gap"]) → {"w": 1.2, "gap": 0.1}
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, xdict):
        return {k: xdict[k] for k in self.keys if k in xdict}
