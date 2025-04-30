import os
import mwlab
from filters import TouchstoneMWFilterDataset, MWFilter
from mwlab.transforms.s_transforms import S_Crop
import matplotlib.pyplot as plt


ENV_DATASET_PATH = os.getcwd()+"\\Data"


def main():
    tds = TouchstoneMWFilterDataset(root=ENV_DATASET_PATH)
    print(f"Загружено файлов: {len(tds)}")
    print("Пример параметров из первого файла:")
    print(tds[0][0], "\n")
    net: MWFilter = tds[0][1]
    y_transform = S_Crop(f_start=net.f0-net.bw*1.2, f_stop=net.f0+net.bw*1.2, unit='MHz')
    tds_transformed = TouchstoneMWFilterDataset(root=ENV_DATASET_PATH, s_tf=y_transform)
    analyzer = mwlab.TouchstoneDatasetAnalyzer(tds_transformed)
    analyzer.plot_s_stats(port_in=2, port_out=1)
    analyzer.plot_s_stats(port_in=1, port_out=1)


if __name__ == "__main__":
    main()
    plt.show()
