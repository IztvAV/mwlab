import numpy as np
import skrf as rf

def make_network(frequencies, S11, S21, S22, name=""):
    n = len(frequencies)
    S = np.zeros((n, 2, 2), dtype=complex)
    S[:, 0, 0] = S11
    S[:, 0, 1] = S21
    S[:, 1, 0] = S21
    S[:, 1, 1] = S22
    ntw = rf.Network(
        frequency=rf.Frequency.from_f(frequencies, unit='Hz'),
        s=S,
        name=name
    )
    return ntw