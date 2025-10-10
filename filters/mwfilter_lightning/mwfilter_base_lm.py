import torch.nn
import torchmetrics

from mwlab import BaseLModule, TouchstoneLDataModule, BaseLMWithMetrics
from filters import MWFilter, CouplingMatrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from filters.mwfilter_optim.base import FastMN2toSParamCalculation


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
        pred_fil = self.create_filter_from_prediction(orig_fil, pred_prms, meta)
        return orig_fil, pred_fil

    @staticmethod
    def create_filter_from_prediction(orig_fil: MWFilter, pred_prms: dict, meta: dict) -> MWFilter:
        # orig_fil = MWFilter.from_touchstone_dataset_item(({**meta['params'], **orig_prms}, orig_fil))
        # Q = pred_prms.pop('Q')
        Q = meta['params']['Q']
        pred_matrix = MWFilter.matrix_from_touchstone_data_parameters({**meta['params'], **pred_prms})
        s_pred = MWFilter.response_from_coupling_matrix(f0=meta['params']['f0'], FBW=meta['params']['bw']/meta['params']['f0'], frange=orig_fil.f / 1e6,
                                                        Q=Q, M=pred_matrix)
        pred_fil = MWFilter(order=int(meta['params']['N']), bw=meta['params']['bw'], f0=meta['params']['f0'],
                            Q=Q,
                            matrix=pred_matrix, frequency=orig_fil.f, s=s_pred, z0=orig_fil.z0)
        return pred_fil

    def plot_origin_vs_prediction(self, origin_fil: MWFilter, pred_fil: MWFilter):
        plt.figure()
        origin_fil.plot_s_db(m=0, n=0, label='S11 origin')
        origin_fil.plot_s_db(m=1, n=0, label='S21 origin')
        origin_fil.plot_s_db(m=1, n=1, label='S22 origin')
        pred_fil.plot_s_db(m=0, n=0, label='S11 pred', ls=':')
        pred_fil.plot_s_db(m=1, n=0, label='S21 pred', ls=':')
        pred_fil.plot_s_db(m=1, n=1, label='S22 pred', ls=':')


class MWFilterBaseLMWithMetricsCAE(MWFilterBaseLMWithMetrics):
    def __init__(self, origin_filter: MWFilter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_filter = origin_filter
        self.fast_calc = FastMN2toSParamCalculation(matrix_order=self.origin_filter.coupling_matrix.matrix_order,
                                                    wlist=self.origin_filter.f_norm, Q=self.origin_filter.Q,
                                                    fbw=self.origin_filter.fbw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """X → Z (учитываем scaler_in, если есть)."""
        if self.scaler_in is not None:
            x = self.scaler_in(x)
        preds = self.model(x)
        matrix_factors = self.scaler_out.inverse(preds)
        s_origin_db = MWFilter.to_db(x)
        matrix = CouplingMatrix.from_factors(matrix_factors, self.origin_filter.coupling_matrix.links,
                                             self.origin_filter.coupling_matrix.matrix_order)
        s_pred = self.fast_calc.RespM2(matrix)
        s_pred_db = MWFilter.to_db(s_pred)
        loss = self.loss_fn(s_pred_db, s_origin_db)
        return loss

    def _shared_step(self, batch):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        return preds
