# transforms/__init__.py
class Compose:
    def __init__(self, tfs): self.tfs = tfs
    def __call__(self, x):
        for tf in self.tfs:
            x = tf(x)
        return x

