import os
import numpy as np
from filters import MWFilter


class CMTheoreticalDatasetGenerator:
    def __init__(self,
                 path_to_save_dataset: str,
                 origin_filter: MWFilter,
                 cm_shifts: np.array,
                 ps_shifts: np.array,
                 ):
        pass