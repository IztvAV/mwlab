import os
import mwlab
from mwlab.transforms.s_transforms import S_Crop, S_Resample
from mwlab.transforms import TComposite

from filters import TouchstoneMWFilterDataset, MWFilter, CouplingMatrix
from filters import CMTheoreticalDatasetGenerator, CMTheoreticalDatasetGeneratorSamplers
from filters import Sampler, SamplerTypes

import matplotlib.pyplot as plt
import numpy as np


ENV_DATASET_PATH = os.getcwd()+"\\Data\\origins_data"


def create_min_max_matrices(origin_matrix: CouplingMatrix, deltas: np.array):
    Mmin = origin_matrix.matrix
    Mmax = origin_matrix.matrix

    for (i, j) in origin_matrix.links:
        if i == j:
            Mmin[i][j] -= deltas[0]
            Mmax[i][j] += deltas[0]
        elif j == (i + 1):
            Mmin[i][j] -= deltas[1]
            Mmin[j][i] -= deltas[1]
            Mmax[i][j] += deltas[1]
            Mmax[j][i] += deltas[1]
        else:
            Mmin[i][j] -= deltas[2]
            Mmin[j][i] -= deltas[2]
            Mmax[i][j] += deltas[2]
            Mmax[j][i] += deltas[2]
    return Mmin, Mmax


def create_min_max_phase_shifts(origin_shifts: np.array, deltas):
    phase_shifts_min = origin_shifts - deltas
    phase_shifts_max = origin_shifts + deltas
    return phase_shifts_min, phase_shifts_max

def main():
    tds = TouchstoneMWFilterDataset(source=ENV_DATASET_PATH)
    print(f"Загружено файлов: {len(tds)}")
    print("Пример параметров из первого файла:")
    print(tds[0][0], "\n")
    net: MWFilter = tds[0][1]
    y_transform = TComposite([
        S_Crop(f_start=net.f0-net.bw*1.2, f_stop=net.f0+net.bw*1.2, unit='MHz'),
        S_Resample(301)
    ])
    tds_transformed = TouchstoneMWFilterDataset(source=ENV_DATASET_PATH, s_tf=y_transform)
    analyzer = mwlab.TouchstoneDatasetAnalyzer(tds_transformed)
    analyzer.plot_s_stats(port_in=2, port_out=1)
    analyzer.plot_s_stats(port_in=1, port_out=1)

    codec = mwlab.TouchstoneCodec.from_dataset(tds_transformed)
    print(codec)
    codec.y_channels = ['S1_1.real', 'S2_1.real', 'S2_2.real', 'S1_1.imag', 'S2_1.imag', 'S2_2.imag']
    print("Каналы:", codec.y_channels)
    print("Количество каналов:", len(codec.y_channels))

    # Пример кодирования и декодирования
    prms, net = tds_transformed[0]  # Используем первый файл набора
    ts = mwlab.TouchstoneData(net, prms)  # Создаем объект TouchstoneData

    # Кодирование
    x, y, meta = codec.encode(ts)
    print("x-параметры:\n", x)
    print("y-параметры:\n", y)
    print("Метаданные:\n" + str(meta))

    # Декодирование
    ts_rec = codec.decode(y, meta)


    m_min, m_max = create_min_max_matrices(origin_matrix=net.coupling_matrix, deltas=np.array([1.5, 0.1, 0.005]))
    phase_shifts_min, phase_shifts_max = create_min_max_phase_shifts(origin_shifts=np.array([0.547, -1.0, 0.01685, 0.017]),
                                                                     deltas=np.array([0.02, 0.02, 0.005, 0.005]))
    samplers = CMTheoreticalDatasetGeneratorSamplers(
        cm_shifts=Sampler.uniform(start=m_min, stop=m_max, num=1000),
        ps_shifts=Sampler.uniform(start=phase_shifts_min, stop=phase_shifts_max, num=1000)
    )

    ds_gen = CMTheoreticalDatasetGenerator(
        path_to_save_dataset=os.getcwd()+"\\Data\\datasets_data",
        origin_filter=net,
        samplers=samplers
    )
    ds_gen.generate()
    pass

if __name__ == "__main__":
    main()
    plt.show()
