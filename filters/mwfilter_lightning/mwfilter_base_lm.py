from typing import Dict

import torch.nn
import torchmetrics

from mwlab import BaseLModule, TouchstoneLDataModule, BaseLMWithMetrics
from filters import MWFilter, CouplingMatrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
import skrf as rf


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
        pred_matrix = MWFilter.matrix_from_touchstone_data_parameters({**meta['params'], **pred_prms})
        s_pred = MWFilter.response_from_coupling_matrix(f0=meta['params']['f0'], FBW=meta['params']['bw']/meta['params']['f0'], frange=orig_fil.f / 1e6,
                                                        Q=meta['params']['Q'], M=pred_matrix)
        pred_fil = MWFilter(order=int(meta['params']['N']), bw=meta['params']['bw'], f0=meta['params']['f0'],
                            Q=meta['params']['Q'],
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


class MWFilterBaseLMWithMetricsAE(MWFilterBaseLMWithMetrics):
    def __init__(self, meta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = meta

    def predict_for(self, dm: TouchstoneLDataModule, idx: int, with_scalers=True) -> tuple[MWFilter, MWFilter]:
        # Возьмем для примера первый touchstone-файл из тестового набора данных
        test_tds = dm.get_dataset(split="test", meta=True)
        # Поскольку swap_xy=True, то датасет меняет местами пары (y, x)
        y_t, x_t, meta = test_tds[idx]  # Используем первый файл набора данных

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

        orig_fil =  rf.Network(frequency=net.f, s=net.s, z0=net.z0)
        pred_fil = dm.codec.decode_s(pred_prms, meta)
        return orig_fil, pred_fil

    @torch.no_grad()
    def predict_x(self, net: rf.Network) -> Dict[str, float]:
        if not self.swap_xy:
            raise RuntimeError("predict_x доступен только при swap_xy=True")
        if self.codec is None:
            raise RuntimeError("predict_x требует codec")
        self.eval()
        y_t, _ = self.codec.encode_s(net)
        y_t = y_t.to(self.device).unsqueeze(0)
        x_pred = self(y_t)[0]

        x_pred = self.scaler_in.inverse(x_pred)
        return x_pred

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """X → Z (учитываем scaler_in, если есть)."""
        if self.scaler_in is not None:
            x = self.scaler_in(x)
        preds, z = self.model(x)
        # loss = self.loss_fn(x, preds)
        return preds

    def _shared_step(self, batch):
        def sparams_to_complex(x: torch.Tensor) -> torch.Tensor:
            """
            x: (B, 8, F) — в заданном порядке каналов.
            return: S (B, 2, 2, F), dtype: complex64/complex128 в зависимости от x.dtype
            """
            assert x.ndim == 3 and x.size(1) == 8, "Ожидаю тензор формы (B, 8, F)"
            B, _, F = x.shape
            re11, re12, re21, re22 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]  # (B, F)
            im11, im12, im21, im22 = x[:, 4], x[:, 5], x[:, 6], x[:, 7]  # (B, F)

            # Собираем комплексные каналы
            S11 = torch.complex(re11, im11)  # (B, F)
            S12 = torch.complex(re12, im12)
            S21 = torch.complex(re21, im21)
            S22 = torch.complex(re22, im22)

            # Укладываем в матрицу 2x2 по портам
            # S[b] =
            # [[S11(b,f), S12(b,f)],
            #  [S21(b,f), S22(b,f)]], f по последней оси
            S = torch.stack([
                torch.stack([S11, S12], dim=1),  # (B, 2, F)
                torch.stack([S21, S22], dim=1)  # (B, 2, F)
            ], dim=1)  # (B, 2, 2, F)
            return S
        x, y, _ = self._split_batch(batch)
        preds_forward = self(x)
        preds_backward = self(torch.flip(x, dims=(-1,)))
        preds_backward = torch.flip(preds_backward, dims=(-1,))
        if self.scaler_in is not None:
            x = self.scaler_in(x)
        loss = self.loss_fn(preds_forward, x) + self.loss_fn(preds_backward, x)
        if self.scaler_in is not None:
            x = self.scaler_in.inverse(x)
            preds_forward = self.scaler_in.inverse(preds_forward)
            preds_backward = self.scaler_in.inverse(preds_backward)
        s_x = sparams_to_complex(x)
        s_preds_forward = sparams_to_complex(preds_forward)
        s_preds_backward = sparams_to_complex(preds_backward)
        loss += 0.1*(self.loss_fn(torch.log10(torch.abs(s_x)+1e-5), torch.log10(torch.abs(s_preds_forward)+1e-5)) + self.loss_fn(torch.log10(torch.abs(s_x)+1e-5), torch.log10(torch.abs(s_preds_backward)+1e-5)))
        return loss

    # ======================================================================
    #                        validation / test loop
    # ======================================================================
    def validation_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        # y_t = self._prepare_targets(y)

        if self.scaler_in is not None:
            x = self.scaler_in(x)
        loss = self.loss_fn(preds, x)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        # 💡 Flatten для совместимости с метриками
        preds_flat = preds.view(preds.size(0), -1)
        y_flat = x.view(x.size(0), -1)

        metric_dict = self.val_metrics(preds_flat, y_flat)
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        # y_t = self._prepare_targets(y)

        if self.scaler_in is not None:
            x = self.scaler_in(x)
        loss = self.loss_fn(preds, x)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        # 💡 Flatten для метрик
        preds_flat = preds.view(preds.size(0), -1)
        y_flat = x.view(x.size(0), -1)

        metric_dict = self.test_metrics(preds_flat, y_flat)
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def plot_origin_vs_prediction(self, origin_fil: MWFilter, pred_fil: MWFilter):
        plt.figure()
        origin_fil.plot_s_db(m=0, n=0, label='S11 origin')
        origin_fil.plot_s_db(m=1, n=0, label='S21 origin')
        origin_fil.plot_s_db(m=1, n=1, label='S22 origin')
        pred_fil.plot_s_db(m=0, n=0, label='S11 pred', ls=':')
        pred_fil.plot_s_db(m=1, n=0, label='S21 pred', ls=':')
        pred_fil.plot_s_db(m=1, n=1, label='S22 pred', ls=':')