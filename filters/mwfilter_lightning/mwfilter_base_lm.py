import torch.nn
from mwlab import BaseLModule, TouchstoneLDataModule, BaseLMWithMetrics
from filters import MWFilter, CouplingMatrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class MWFilterBaseLModule(BaseLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def predict_for(self, dm: TouchstoneLDataModule, idx: int) -> tuple[MWFilter, MWFilter]:
        # Возьмем для примера первый touchstone-файл из тестового набора данных
        test_tds = dm.get_dataset(split="test", meta=True)
        # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
        y_t, x_t, meta = test_tds[idx]  # Используем первый файл набора данных]

        # Декодируем данные
        orig_prms = dm.codec.decode_x(x_t)  # Создаем словарь параметров
        net = dm.codec.decode_s(y_t, meta)  # Создаем объект skrf.Network

        # Предсказанные S-параметры
        pred_prms = self.predict_x(net)

        print(f"Исходные параметры: {orig_prms}")
        print(f"Предсказанные параметры: {pred_prms}")

        orig_fil = MWFilter.from_touchstone_dataset_item(({**meta['params'], **orig_prms}, net))
        pred_matrix = MWFilter.matrix_from_touchstone_data_parameters({**meta['params'], **pred_prms})
        s_pred = MWFilter.response_from_coupling_matrix(f0=orig_fil.f0, FBW=orig_fil.fbw, frange=orig_fil.f/1e6,
                                                        Q=orig_fil.Q, M=pred_matrix)
        pred_fil = MWFilter(order=int(meta['params']['N']), bw=meta['params']['bw'], f0=meta['params']['f0'], Q=meta['params']['Q'],
                 matrix=pred_matrix, frequency=orig_fil.f, s=s_pred, z0=orig_fil.z0)
        return orig_fil, pred_fil

    def plot_origin_vs_prediction(self, origin_fil: MWFilter, pred_fil: MWFilter):
        plt.figure()
        origin_fil.plot_s_db(m=0, n=0, label='S11 origin')
        origin_fil.plot_s_db(m=1, n=0, label='S21 origin')
        pred_fil.plot_s_db(m=0, n=0, label='S11 pred', ls=':')
        pred_fil.plot_s_db(m=1, n=0, label='S21 pred', ls=':')


class MWFilterBaseLMWithMetrics(BaseLMWithMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, dm: TouchstoneLDataModule, idx: int, with_scalers: bool=True):
        if idx == -1:  # значит предсказываем всем датасете
            predictions = [self.predict_for(dm, i, with_scalers) for i in range(len(dm.get_dataset(split="test", meta=True)))]
        else:
            predictions = self.predict_for(dm, idx, with_scalers)
        return predictions

    def predict_for(self, dm: TouchstoneLDataModule, idx: int, with_scalers=True) -> tuple[MWFilter, MWFilter]:
        # Возьмем для примера первый touchstone-файл из тестового набора данных
        test_tds = dm.get_dataset(split="test", meta=True)
        # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
        y_t, x_t, meta = test_tds[idx]  # Используем первый файл набора данных]

        # Декодируем данные
        orig_prms = dm.codec.decode_x(x_t)  # Создаем словарь параметров
        net = dm.codec.decode_s(y_t, meta)  # Создаем объект skrf.Network

        # Предсказанные S-параметры
        pred_prms = self.predict_x(net)
        if not with_scalers:
            pred_prms_vals = dm.scaler_out(torch.tensor(list(pred_prms.values())))
            orig_prms_vals = dm.scaler_out(torch.tensor(list(orig_prms.values())))
            pred_prms = dict(zip(pred_prms.keys(), list(torch.squeeze(pred_prms_vals, dim=0).numpy())))
            orig_prms = dict(zip(orig_prms.keys(), list(torch.squeeze(orig_prms_vals, dim=0).numpy())))

        print(f"Исходные параметры: {orig_prms}")
        print(f"Предсказанные параметры: {pred_prms}")

        orig_fil = MWFilter.from_touchstone_dataset_item(({**meta['params'], **orig_prms}, net))
        pred_matrix = MWFilter.matrix_from_touchstone_data_parameters({**meta['params'], **pred_prms})
        s_pred = MWFilter.response_from_coupling_matrix(f0=orig_fil.f0, FBW=orig_fil.fbw, frange=orig_fil.f / 1e6,
                                                        Q=orig_fil.Q, M=pred_matrix)
        pred_fil = MWFilter(order=int(meta['params']['N']), bw=meta['params']['bw'], f0=meta['params']['f0'],
                            Q=meta['params']['Q'],
                            matrix=pred_matrix, frequency=orig_fil.f, s=s_pred, z0=orig_fil.z0)
        return orig_fil, pred_fil

    """ Функция для вычисления MSELoss от списка полученных фильтров. Нужна для оценки предсказания модели на наборе данных """
    def mse_score(self, predictions: list[tuple[MWFilter, MWFilter]]):
        score = 0.0
        loss_f = torch.nn.MSELoss()
        for orig_fil, pred_fil in predictions:
            loss = loss_f(torch.tensor(orig_fil.coupling_matrix.factors), torch.tensor(pred_fil.coupling_matrix.factors))
            score += loss.item()
        score /= len(predictions)
        return score

    def plot_origin_vs_prediction(self, origin_fil: MWFilter, pred_fil: MWFilter):
        plt.figure()
        origin_fil.plot_s_db(m=0, n=0, label='S11 origin')
        origin_fil.plot_s_db(m=1, n=0, label='S21 origin')
        pred_fil.plot_s_db(m=0, n=0, label='S11 pred', ls=':')
        pred_fil.plot_s_db(m=1, n=0, label='S21 pred', ls=':')