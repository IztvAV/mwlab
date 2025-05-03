import os
import mwlab
from filters import TouchstoneMWFilterDataset, MWFilter
from mwlab.transforms.s_transforms import S_Crop, S_Resample
from mwlab.transforms import TComposite
import matplotlib.pyplot as plt


ENV_DATASET_PATH = os.getcwd()+"\\Data"


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
    codec.y_channels = ['S11.real', 'S21.real', 'S22.real', 'S11.imag', 'S21.imag', 'S22.imag']
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
    pass

if __name__ == "__main__":
    main()
    plt.show()
