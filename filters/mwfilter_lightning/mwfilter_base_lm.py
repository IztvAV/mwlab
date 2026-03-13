import re

import torch.nn
import torchmetrics

from mwlab import BaseLModule, TouchstoneLDataModule, BaseLMWithMetrics, TouchstoneCodec
from filters import MWFilter, CouplingMatrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
from mwlab.lightning.base_lm_with_metrics import MAE_error
import torch.nn.functional as F
import re
from contextlib import contextmanager
from typing import Optional, Dict, Iterator

import torch
import torch.nn.functional as F


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


def batched_sparams_from_M(M, w_norm, Q, fbw, a11, a22, b11, b22):
    """
    M:     [B, N, N] complex64 — матрицы связи (реальные значения в Re)
    w_norm:[F] float32         — нормированная сетка ω (как у вашего решателя)
    Q:     [B] float32
    fbw:   [B] float32
    a11,a22,b11,b22: [B] float32 — фазовые параметры θ1=a11+b11*w, θ2=a22+b22*w (рад)
    Возвращает:
      S: [B, F, 2, 2] complex64  (S11,S21,S12=S21,S22) — уже с фазовыми множителями.
    """

    device = M.device
    B, N, _ = M.shape
    F = int(w_norm.numel())

    # --- R и I (как в вашем FastMN2toSParamCalculation)
    R = torch.zeros(N, N, dtype=torch.complex64, device=device)
    R[0, 0] = 1j
    R[-1, -1] = 1j

    I = torch.eye(N, N, dtype=torch.complex64, device=device)
    I[0, 0] = 0
    I[-1, -1] = 0

    # маска диагонали для внутренних резонаторов (под G)
    Gmask = torch.eye(N, N, dtype=torch.complex64, device=device)
    Gmask[0, 0] = 0
    Gmask[-1, -1] = 0

    # --- Матрица A(ω) = (M-R) + ω I - G(Q,fbw)
    MR = (M - R).unsqueeze(1)  # [B, 1, N, N]
    w_calc = w_norm.view(1, F, 1, 1).to(device)  # [1, F, 1, 1]

    coeff = (1.0 / (fbw * Q)).to(torch.float32).to(device)  # [B] real
    G_b = (1j * coeff).view(B, 1, 1, 1).to(torch.complex64) * Gmask.view(1, 1, N, N)  # [B,1,N,N]

    A = MR + w_calc * I.view(1, 1, N, N) - G_b  # [B, F, N, N]

    # --- RHS: e0 и eN
    rhs = torch.zeros(B, F, N, 2, dtype=torch.complex64, device=device)
    rhs[:, :, 0, 0] = 1.0
    rhs[:, :, -1, 1] = 1.0

    # Решаем A X = RHS для двух правых частей
    # torch.linalg.solve работает батчево по первым размерностям
    X = torch.linalg.solve(A, rhs)  # [B, F, N, 2]

    x0 = X[..., 0]  # [B, F, N] — A^{-1} @ e0
    xN = X[..., 1]  # [B, F, N] — A^{-1} @ eN

    A00 = x0[..., 0]  # [B, F]
    AN0 = x0[..., -1]  # [B, F]
    ANN = xN[..., -1]  # [B, F]

    S11 = 1 + 2j * A00
    S21 = -2j * AN0
    S22 = 1 + 2j * ANN

    # --- Фазовые множители индивидуально для каждого образца
    # θ1 = a11 + b11*w, θ2 = a22 + b22*w
    w = w_norm.view(1, F).to(device)
    phi1 = a11.view(B, 1) + b11.view(B, 1) * w  # [B, F]
    phi2 = a22.view(B, 1) + b22.view(B, 1) * w  # [B, F]

    f11 = torch.exp(-1j * 2.0 * phi1).to(torch.complex64)  # [B, F]
    f22 = torch.exp(-1j * 2.0 * phi2).to(torch.complex64)  # [B, F]
    f21 = torch.exp(-1j * (phi1 + phi2)).to(torch.complex64)  # [B, F]

    S11 = S11 * f11
    S21 = S21 * f21
    S22 = S22 * f22

    # Собираем S матрицы
    S = torch.zeros(B, F, 2, 2, dtype=torch.complex64, device=device)
    S[:, :, 0, 0] = S11
    S[:, :, 1, 1] = S22
    S[:, :, 0, 1] = S21
    S[:, :, 1, 0] = S21
    return S


def unpack_batch_predictions(preds, scaler_out, names, N, device=None, dtype=torch.float32):
    """
       preds: [B, D] — предсказания в шкале модели
       names: список имён (длина D). Имя m_i_j -> элемент матрицы связи (i,j).
       N: порядок полной матрицы связи (включая 2 порта), по вашим именам у вас N=12.
       Возвращает:
         M:   [B, N, N] complex64 — симметричная (реальная часть), готова к решателю
         Q:   [B]      float32
         fbw: [B]      float32   (долевая полоса, FBW)
         f0:  [B]      float32   (Гц, если так обучали)
         a11,a22,b11,b22: [B] float32 — фазовые параметры (рад)
       """
    if device is None: device = preds.device
    preds = scaler_out.inverse(preds).to(device=device, dtype=dtype)  # [B, D]
    B, D = preds.shape

    # Подготовим M как real и потом приведём к complex
    M_real = torch.zeros(B, N, N, dtype=dtype, device=device)

    # Плейсхолдеры для скалярных выходов
    out_vals = {k: torch.zeros(B, dtype=dtype, device=device) for k in
                ['Q', 'f0', 'bw', 'a11', 'a22', 'b11', 'b22']}

    # Предкомпилируем регэксп для m_i_j
    pat = re.compile(r'^m_(\d+)_(\d+)$')

    for col, name in enumerate(names):
        m = pat.match(name)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            v = preds[:, col]  # [B]
            M_real[:, i, j] = v
            M_real[:, j, i] = v  # симметрия
        else:
            if name in out_vals:  # Q, bw, f0, a11, a22, b11, b22
                out_vals[name] = preds[:, col]
            else:
                # игнорируем неизвестные имена (или бросаем исключение)
                pass

    # Комплексная матрица связи (реальную часть кладём в .real)
    M = M_real.to(torch.complex64)  # им. часть == 0, тип совместим с решателем

    return (M,
            out_vals['Q'],
            out_vals['f0'],
            out_vals['bw'],
            out_vals['a11'], out_vals['a22'], out_vals['b11'], out_vals['b22'])


def sparams_from_preds_batch(preds, scaler_out, names, N, w_norm, device=None):
    """
    preds -> (M,Q,fbw,f0,фазы) -> S (батч).
    Возвращает:
      S: [B, F, 2, 2] complex64
      aux: словарь с Q, fbw, f0 и т.п.
    """
    if device is None: device = preds.device
    (M, Q, f0, bw, a11, a22, b11, b22) = unpack_batch_predictions(
        preds, scaler_out, names, N, device=device, dtype=torch.float32
    )
    w_norm = torch.tensor(w_norm, dtype=torch.float32, device=device)

    fbw = bw/f0
    S = batched_sparams_from_M(M, w_norm, Q, fbw, a11, a22, b11, b22)

    aux = {'Q': Q, 'f0': f0, 'bw': bw, 'a11': a11, 'a22': a22, 'b11': b11, 'b22': b22}
    return S, aux

def sbatch_to_features(S_batched: torch.Tensor,
                       order=None,
                       flatten: str = 'none'):
    """
    S_batched : [B, F, 2, 2] complex
    order     : список имён каналов (см. DEFAULT_ORDER). Если None — DEFAULT_ORDER.
    flatten   : 'none' -> [B, F, C]
                'bf_c' -> [B*F, C]
                'b_fc' -> [B, F*C]
    Возвращает:
      feats : float32 тензор с указанной формой
      idx_map : словарь {имя_канала: индекс_канала} — удобно для дебага
    """
    B, F, _, _ = S_batched.shape

    feat_map = {
        'S1_1.real': S_batched[..., 0, 0].real,
        'S1_2.real': S_batched[..., 0, 1].real,
        'S2_1.real': S_batched[..., 1, 0].real,
        'S2_2.real': S_batched[..., 1, 1].real,
        'S1_1.imag': S_batched[..., 0, 0].imag,
        'S1_2.imag': S_batched[..., 0, 1].imag,
        'S2_1.imag': S_batched[..., 1, 0].imag,
        'S2_2.imag': S_batched[..., 1, 1].imag,
    }

    # Стек по оси каналов -> [B, C, F]
    feats = torch.stack([feat_map[name].to(torch.float32) for name in order], dim=1)
    idx_map = {name: i for i, name in enumerate(order)}
    return feats, idx_map

def features_to_sbatch(
        feats: torch.Tensor,
        order=None,
        *,
        layout: str = 'BCF',  # 'BCF' | 'BFC' | 'auto'  (по умолчанию ожидаем [B,C,F])
        freq_len: int = None,  # нужно для 2D входов
        dtype=torch.complex64
):
    """
    Преобразует фичи каналов в батч S-параметров.

    feats  : тензор фич:
             - [B, C, F] при layout='BCF' (по умолчанию)
             - [B, F, C] при layout='BFC'
             - [B*F, C]  или [B, F*C] при layout='auto' (нужен freq_len)
    order  : список имён каналов (см. DEFAULT_ORDER). Если None — DEFAULT_ORDER.
    layout : схема входа. При 'auto' пытаемся угадать, при 2D необходим freq_len.
    freq_len: длина частотной оси F (для 2D входов).
    dtype  : тип комплексного выхода (по умолчанию complex64).

    Возвращает:
      S : [B, F, 2, 2] complex
      idx_map : {имя_канала: индекс_канала}
    """
    C_expected = len(order)

    x = feats
    dev = x.device
    # ---- Нормализуем к [B, F, C] ----
    if x.ndim == 3:
        B, A, B_or_C = x.shape
        if layout.upper() == 'BCF' or (layout == 'auto' and A == C_expected):
            # [B, C, F] -> [B, F, C]
            x = x.transpose(1, 2).contiguous()
        elif layout.upper() == 'BFC' or (layout == 'auto' and B_or_C == C_expected):
            # [B, F, C] как есть
            pass
        else:
            raise ValueError(f"Не могу определить раскладку 3D входа {tuple(feats.shape)} при layout='{layout}'.")
    elif x.ndim == 2:
        if freq_len is None:
            raise ValueError("Для 2D входа укажите freq_len.")
        N0, C_in = x.shape
        if C_in == C_expected:
            # [B*F, C] -> [B, F, C]
            if N0 % freq_len != 0:
                raise ValueError(f"BF ({N0}) не делится на F ({freq_len}).")
            B = N0 // freq_len
            x = x.view(B, freq_len, C_expected)
        elif C_in % C_expected == 0 and N0 > 0 and N0 == N0:  # возможно [B, F*C]
            # [B, F*C] -> [B, F, C]
            F = freq_len
            B = N0
            if C_in != F * C_expected:
                raise ValueError(f"Ожидалось F*C = {F * C_expected}, а пришло {C_in}.")
            x = x.view(B, F, C_expected)
        else:
            raise ValueError(
                f"2D вход {tuple(feats.shape)} не распознан. Ожидаю последнюю размерность = C ({C_expected}) "
                f"или кратную F*C (freq_len нужен).")
    else:
        raise ValueError("Поддерживаются входы 2D или 3D.")

    # x сейчас [B, F, C]
    B, F, C = x.shape
    if C != C_expected:
        raise ValueError(f"Число каналов C={C} не совпадает с длиной order={C_expected}.")

    # Словарь каналов: имя -> [B, F]
    chans = {name: x[..., i] for i, name in enumerate(order)}

    # Утилита: получить канал или нули (если вдруг канала нет)
    def get(name):
        if name in chans:
            return chans[name]
        # если каких-то каналов нет, заполним нулями
        return torch.zeros(B, F, device=dev, dtype=x.dtype)

    # Собираем комплексные S
    S11 = get('S1_1.real').to(torch.float32) + 1j * get('S1_1.imag').to(torch.float32)
    S12 = get('S1_2.real').to(torch.float32) + 1j * get('S1_2.imag').to(torch.float32)
    S21 = get('S2_1.real').to(torch.float32) + 1j * get('S2_1.imag').to(torch.float32)
    S22 = get('S2_2.real').to(torch.float32) + 1j * get('S2_2.imag').to(torch.float32)

    S = torch.zeros(B, F, 2, 2, dtype=dtype, device=dev)
    S[..., 0, 0] = S11.to(dtype)
    S[..., 0, 1] = S12.to(dtype)
    S[..., 1, 0] = S21.to(dtype)
    S[..., 1, 1] = S22.to(dtype)

    idx_map = {name: i for i, name in enumerate(order)}
    return S, idx_map

def wrap_pm_pi(phi: torch.Tensor) -> torch.Tensor:
    twopi = 2 * torch.pi
    return (phi + torch.pi) % twopi - torch.pi

def remove_port_phase_shift(s_b: torch.Tensor, f_norm: torch.Tensor,
                            a11: torch.Tensor, b11: torch.Tensor,
                            a22: torch.Tensor, b22: torch.Tensor,
                            *, canonicalize: bool = False) -> torch.Tensor:
    B, F, _, _ = s_b.shape

    # -> (B,F)
    if f_norm.dim() == 1:
        f_bf = f_norm.view(1, F).expand(B, F)
    elif f_norm.dim() == 2 and f_norm.shape[0] == 1:
        f_bf = f_norm.expand(B, F)
    else:
        f_bf = f_norm  # (B,F)

    # ---- Канонизация a по опорной точке ----
    if canonicalize:
        # 1) простой канон: a в (-pi, pi]
        a11 = wrap_pm_pi(a11)
        a22 = wrap_pm_pi(a22)

        # 2) дополнительный (опционально): якорим phi в центре сетки (если хотите максимально стабильно)
        # idx0 = F // 2
        # f0 = f_bf[:, idx0]  # (B,)
        # phi1_0 = a11 + b11 * f0
        # phi2_0 = a22 + b22 * f0
        # a11 = a11 - torch.round(phi1_0 / (2*torch.pi)) * (2*torch.pi)
        # a22 = a22 - torch.round(phi2_0 / (2*torch.pi)) * (2*torch.pi)

    phi1 = a11.view(B, 1) + b11.view(B, 1) * f_bf
    # phi1 = b11.view(B, 1) * f_bf
    phi2 = a22.view(B, 1) + b22.view(B, 1) * f_bf
    # phi2 = b22.view(B, 1) * f_bf

    e1 = torch.exp(1j * phi1)
    e2 = torch.exp(1j * phi2)

    s_de = s_b.clone()
    s_de[:, :, 0, :] *= e1.unsqueeze(-1)
    s_de[:, :, 1, :] *= e2.unsqueeze(-1)
    s_de[:, :, :, 0] *= e1.unsqueeze(-1)
    s_de[:, :, :, 1] *= e2.unsqueeze(-1)
    return s_de


class MWFilterBaseLMWithMetrics(BaseLMWithMetrics):
    def __init__(self, work_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.work_model = work_model

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
        pred_fil = self.create_filter_from_prediction(orig_fil, self.work_model.orig_filter, pred_prms, self.codec)
        return orig_fil, pred_fil

    @staticmethod
    def create_filter_from_prediction(orig_fil: MWFilter, work_model_orig_fil: MWFilter, pred_prms: dict, codec: TouchstoneCodec) -> MWFilter:
        # Q = meta['params']['Q']
        Q = work_model_orig_fil.Q
        pred_matrix = MWFilter.matrix_from_touchstone_data_parameters(pred_prms, matrix_order=work_model_orig_fil.coupling_matrix.matrix_order)
        s_pred = MWFilter.response_from_coupling_matrix(f0=work_model_orig_fil.f0, FBW=work_model_orig_fil.fbw, frange=orig_fil.f / 1e6,
                                                        Q=Q, M=pred_matrix)
        pred_fil = MWFilter(order=work_model_orig_fil.coupling_matrix.matrix_order-2, bw=work_model_orig_fil.bw, f0=work_model_orig_fil.f0,
                            Q=Q,
                            matrix=pred_matrix, frequency=orig_fil.f, s=s_pred, z0=orig_fil.z0)
        return pred_fil

    def plot_origin_vs_prediction(self, origin_fil: MWFilter, pred_fil: MWFilter, title=None):
        plt.figure(figsize=(4, 3))
        origin_fil.plot_s_db(m=0, n=0, label='S11 origin')
        origin_fil.plot_s_db(m=1, n=0, label='S21 origin')
        origin_fil.plot_s_db(m=1, n=1, label='S22 origin')
        pred_fil.plot_s_db(m=0, n=0, label='S11 pred', ls=':')
        pred_fil.plot_s_db(m=1, n=0, label='S21 pred', ls=':')
        pred_fil.plot_s_db(m=1, n=1, label='S22 pred', ls=':')
        if title:
            plt.title(title)

    # # ---------------------------------------------------------------- forward
    # def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
    #     """X → Z (учитываем scaler_in, если есть)."""
    #     if self.scaler_in is not None:
    #         x = self.scaler_in(x)
    #     return self.model(x)

    # ───────────────────────────────────────────── shared step (train/val/test)
    def _shared_step(self, batch):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        # целевое значение всегда приводим к той же шкале, что и модель
        if self.scaler_out is not None:
            y = self.scaler_out(y)

        loss = self.calc_loss(x, y, preds)
        return loss

    # ======================================================================
    #                        validation / test loop
    # ======================================================================
    def validation_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        y_t = self._prepare_targets(y)

        loss = self.calc_loss(x, y_t, preds)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        # 💡 Flatten для совместимости с метриками
        preds_flat = preds.view(preds.size(0), -1)
        y_flat = y_t.view(y_t.size(0), -1)

        metric_dict = self.val_metrics(preds_flat, y_flat)
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        acc = MAE_error(preds, y_t)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss


class MWFilterBaseLMWithMetricsCM(MWFilterBaseLMWithMetrics):
    def __init__(self, work_model, *args, **kwargs):
        super().__init__(work_model, *args, **kwargs)

    def calc_loss(self, x, y_t, preds):
        s_pred, aux = sparams_from_preds_batch(preds, self.scaler_out, self.work_model.codec.x_keys,
                                               self.work_model.orig_filter.coupling_matrix.matrix_order,
                                               self.work_model.orig_filter.f_norm)

        s_pred_encoded, _ = sbatch_to_features(s_pred, order=self.work_model.codec.y_channels)
        s_in_decoded, _ = features_to_sbatch(x, order=self.work_model.codec.y_channels)

        # s_pred_norm = self.scaler_in(s_pred_encoded)

        loss = (self.loss_fn(preds, y_t) + 0.1*torch.nn.functional.l1_loss(s_pred_encoded, x)
                + 0.05 * torch.nn.functional.l1_loss(torch.angle(s_pred), torch.angle(s_in_decoded)) + (0.01 * torch.abs(preds-y_t).max() + 1e-5 * torch.abs(s_pred - s_in_decoded).max())*0)
        return loss

    def calc_mirror_loss(self, x: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        Делает "зеркальный" проход и считает loss ТОЧНО так же, как calc_loss, но:
          1) входные данные x переставляются местами по портам -> x_mir
          2) x_mir подаётся в сеть -> preds_mir
          3) preds_mir возвращается в исходный порядок выходов -> preds_mir_un
          4) loss считается по той же формуле, но сравнение идёт с исходными (x, y_t),
             а не с x_mir.

        Важно (про inplace):
          - loss_fn и физический кусок (через sparams_from_preds_batch, где может быть inverse/inplace)
            используют РАЗНЫЕ тензоры preds, чтобы не ловить version mismatch в autograd.
        """

        # ----------------------------------------------------------
        # 1) Зеркалим вход x по портам БЕЗ inplace (через permutation)
        # ----------------------------------------------------------
        y_order = self.work_model.codec.y_channels
        idx_y = {name: i for i, name in enumerate(y_order)}

        required = [
            'S1_1.real', 'S1_1.imag', 'S2_2.real', 'S2_2.imag',
            'S1_2.real', 'S1_2.imag', 'S2_1.real', 'S2_1.imag'
        ]
        if not all(k in idx_y for k in required):
            raise ValueError("Не хватает каналов для зеркалирования входа x (S11/S22/S12/S21 real/imag).")

        C = x.shape[1]
        perm = torch.arange(C, device=x.device)

        # swap S11 <-> S22
        perm[idx_y['S1_1.real']], perm[idx_y['S2_2.real']] = perm[idx_y['S2_2.real']].clone(), perm[
            idx_y['S1_1.real']].clone()
        perm[idx_y['S1_1.imag']], perm[idx_y['S2_2.imag']] = perm[idx_y['S2_2.imag']].clone(), perm[
            idx_y['S1_1.imag']].clone()

        # swap S12 <-> S21
        perm[idx_y['S1_2.real']], perm[idx_y['S2_1.real']] = perm[idx_y['S2_1.real']].clone(), perm[
            idx_y['S1_2.real']].clone()
        perm[idx_y['S1_2.imag']], perm[idx_y['S2_1.imag']] = perm[idx_y['S2_1.imag']].clone(), perm[
            idx_y['S1_2.imag']].clone()

        x_mir = x.index_select(dim=1, index=perm)

        # ----------------------------------------------------------
        # 2) Прогоняем сеть на зеркальном входе
        # ----------------------------------------------------------
        preds_mir = self(x_mir)
        # [B, D]

        # ----------------------------------------------------------
        # 3) "Раззеркаливаем" preds_mir обратно в исходный порядок x_keys
        #    (делаем это перестановкой колонок)
        # ----------------------------------------------------------
        x_keys = self.work_model.codec.x_keys
        idx_x = {name: i for i, name in enumerate(x_keys)}

        K = self.work_model.orig_filter.coupling_matrix.matrix_order  # размер полной матрицы (N+2)
        pat = re.compile(r'^m_(\d+)_(\d+)$')

        src_cols = []
        for name in x_keys:
            m = pat.match(name)
            if m:
                i, j = int(m.group(1)), int(m.group(2))
                ii, jj = (K - 1 - i), (K - 1 - j)
                src_name = f"m_{ii}_{jj}"
            elif name == 'a11':
                src_name = 'a22'
            elif name == 'a22':
                src_name = 'a11'
            elif name == 'b11':
                src_name = 'b22'
            elif name == 'b22':
                src_name = 'b11'
            else:
                src_name = name  # Q, f0, bw и т.п. считаем инвариантными

            if src_name not in idx_x:
                # fallback: если матрица хранится только одним треугольником
                if m:
                    src_alt = f"m_{jj}_{ii}"
                    src_name = src_alt if src_alt in idx_x else name
                else:
                    src_name = name

            src_cols.append(idx_x[src_name])

        src_cols_t = torch.tensor(src_cols, device=preds_mir.device, dtype=torch.long)
        preds_mir_un = preds_mir.index_select(dim=1, index=src_cols_t)

        # ----------------------------------------------------------
        # 4) Считаем loss тем же способом, что в calc_loss, но vs исходных (x, y_t)
        # ----------------------------------------------------------
        # Важно: s_in_decoded берём от исходного x, как ты и хотел
        s_in_decoded, _ = self.features_to_sbatch(x, order=self.work_model.codec.y_channels)

        # Разводим тензоры, чтобы inverse()/прочие inplace не ломали backward loss_fn
        preds_for_loss = preds_mir_un.clone()
        preds_for_phys = preds_mir_un.clone()

        s_pred, _aux = self.sparams_from_preds_batch(
            preds_for_phys,
            self.work_model.codec.x_keys,
            self.work_model.orig_filter.coupling_matrix.matrix_order,
            self.work_model.orig_filter.f_norm
        )
        s_pred_encoded, _ = self.sbatch_to_features(s_pred, order=self.work_model.codec.y_channels)

        loss_mirror = (
                self.loss_fn(preds_for_loss, y_t)
                + 0.1 * F.l1_loss(s_pred_encoded, x)
                + 0.05 * F.l1_loss(torch.angle(s_pred), torch.angle(s_in_decoded))
                + 0.1 *torch.abs(preds_for_loss - y_t).max() + 0.01 * torch.abs(s_pred - s_in_decoded).max()
        )

        return loss_mirror


class MWFilterBaseLMWithMetricsPhaseFreqDep(MWFilterBaseLMWithMetrics):
    def __init__(self, work_model, *args, **kwargs):
        super().__init__(work_model, *args, **kwargs)

    def calc_loss(self, x, y_t, preds):
        x = self.scaler_in.inverse(x)
        s_in_decoded, _ = features_to_sbatch(x, order=self.work_model.codec.y_channels)
        # частотная ось (нормированная)
        f_norm = self.work_model.orig_filter.f_norm  # (F,)
        f_norm = f_norm.to(device=s_in_decoded.device, dtype=torch.float32)

        preds = self.scaler_out.inverse(preds)
        y_t = self.scaler_out.inverse(y_t)

        # распаковка предсказаний
        b11_p = preds[:, 0]
        b22_p = preds[:, 1]
        a11_p = torch.zeros_like(b11_p)
        a22_p = torch.zeros_like(b22_p)

        # удалить фазовый сдвиг
        s_de_p = remove_port_phase_shift(s_in_decoded, f_norm, a11=a11_p, b11=b11_p, a22=a22_p, b22=b22_p)

        b11_t = y_t[:, 0]
        b22_t = y_t[:, 1]
        a11_t = torch.zeros_like(b11_t)
        a22_t = torch.zeros_like(b22_t)

        s_de_t = remove_port_phase_shift(s_in_decoded, f_norm, a11=a11_t, b11=b11_t, a22=a22_t, b22=b22_t)

        # целевое значение всегда приводим к той же шкале, что и модель
        if self.scaler_out is not None:
            y_t = self.scaler_out(y_t)
            preds = self.scaler_out(preds)

        s_de_t_feat, _ = sbatch_to_features(s_de_t, self.work_model.codec.y_channels)
        s_de_p_feat, _ = sbatch_to_features(s_de_p, self.work_model.codec.y_channels)

        if self.scaler_in is not None:
            s_de_t_feat = self.scaler_in(s_de_t_feat)
            s_de_p_feat = self.scaler_in(s_de_p_feat)

        s_de_t_decoded_norm, _ = features_to_sbatch(s_de_t_feat, order=self.work_model.codec.y_channels)
        s_de_p_decoded_norm, _ = features_to_sbatch(s_de_p_feat, order=self.work_model.codec.y_channels)
        eps = 1e-12
        z1 = s_de_p / (s_de_p.abs() + eps)
        z2 = s_de_t / (s_de_t.abs() + eps)
        loss_phase = (1.0 - (z1 * z2.conj()).real).mean()

        loss = self.loss_fn(preds,
                            y_t) + 0.1 * loss_phase + 0.1 * torch.nn.functional.huber_loss(s_de_t_feat, s_de_p_feat)
        # # loss = (self.loss_fn(preds, y_t) + 0.1*torch.nn.functional.l1_loss(s_pred_encoded, x)
        # #         + 0.05 * torch.nn.functional.l1_loss(torch.angle(s_pred), torch.angle(s_in_decoded)) + (0.01 * torch.abs(preds-y_t).max() + 1e-5 * torch.abs(s_pred - s_in_decoded).max())*0)
        # loss = self.loss_fn(preds, y_t)
        return loss


class MWFilterBaseLMWithMetricsPhaseConst(MWFilterBaseLMWithMetrics):
    def __init__(self, work_model, *args, **kwargs):
        super().__init__(work_model, *args, **kwargs)

    def calc_loss(self, x, y_t, preds):
        loss = self.loss_fn(preds, y_t)
        return loss


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, *, device=None, track_buffers: bool = True):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.track_buffers = bool(track_buffers)
        self._ema_state: Dict[str, torch.Tensor] = {}
        self._init_from_model(model, device=device)

    @staticmethod
    def _is_ema_dtype(dtype: torch.dtype) -> bool:
        # EMA имеет смысл для вещественных/комплексных
        return dtype.is_floating_point or dtype.is_complex

    def _init_from_model(self, model: torch.nn.Module, device=None) -> None:
        state = model.state_dict()
        self._ema_state = {}

        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                continue

            t = v.detach()
            if device is not None:
                t = t.to(device=device)

            # Храним ВСЕ тензоры, но EMA будем делать только для float/complex
            self._ema_state[k] = t.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        d = self.decay

        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                continue

            v_det = v.detach()

            if k not in self._ema_state:
                self._ema_state[k] = v_det.clone()
                continue

            ema_v = self._ema_state[k]

            # приводим вход к устройству EMA
            if v_det.device != ema_v.device:
                v_det = v_det.to(device=ema_v.device)

            # 1) float/complex -> EMA
            if self._is_ema_dtype(ema_v.dtype) and self._is_ema_dtype(v_det.dtype):
                # совместим dtype (например fp16 vs fp32)
                if v_det.dtype != ema_v.dtype:
                    v_det = v_det.to(dtype=ema_v.dtype)
                ema_v.mul_(d).add_(v_det, alpha=(1.0 - d))

            # 2) int/bool/long -> просто копируем последнее значение
            else:
                # важно: сохраняем dtype ema_v, но копируем значения
                if v_det.dtype != ema_v.dtype:
                    v_det = v_det.to(dtype=ema_v.dtype)
                ema_v.copy_(v_det)

    @contextmanager
    def swap_to_ema(self, model: torch.nn.Module, *, strict: bool = False) -> Iterator[None]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        model.load_state_dict(self._ema_state, strict=strict)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=strict)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self._ema_state.items()}

    def load_state_dict(self, ema_state: dict[str, torch.Tensor]) -> None:
        self._ema_state = {k: v.detach().clone() for k, v in ema_state.items() if isinstance(v, torch.Tensor)}

    def copy_to(self, model: torch.nn.Module, *, strict: bool = False) -> None:
        model.load_state_dict(self._ema_state, strict=strict)


class MWFilterBaseLModuleEMA(MWFilterBaseLMWithMetrics):
    def __init__(self, work_model, *args, **kwargs):
        super().__init__(work_model, *args, **kwargs)

        self.log_raw_and_ema = False
        self.use_ema = True
        self.ema_decay = 0.999
        self.ema_track_buffers = True
        self.use_ema_on_val = True

        # ✅ ключевой флаг: при загрузке автоматически применять EMA
        self.prefer_ema_weights = True

        self.ema: EMAModel | None = None

    def _ema_target(self) -> torch.nn.Module:
        # EMA ведём только по нейросети
        if not hasattr(self, "model") or self.model is None:
            raise AttributeError("MWFilterBaseLModuleEMA ожидает, что у модуля есть поле self.model (nn.Module).")
        return self.model

    # ---------- hooks ----------
    def on_fit_start(self) -> None:
        if self.use_ema and self.ema is None:
            self.ema = EMAModel(self.model, decay=float(self.ema_decay), track_buffers=bool(self.ema_track_buffers))

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.use_ema and self.ema is not None:
            self.ema.update(self.model)

        # ---------------- checkpoint I/O ----------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # сохраняем EMA отдельно
        if self.use_ema and self.ema is not None:
            checkpoint["ema"] = {
                "decay": float(self.ema_decay),
                "track_buffers": bool(self.ema_track_buffers),
                "state": self.ema.state_dict(),
            }
            checkpoint["prefer_ema_weights"] = bool(self.prefer_ema_weights)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema_pack = checkpoint.get("ema", None)
        if ema_pack is None:
            return

        # создаём EMA объект, если нужно
        if self.ema is None:
            self.ema_decay = float(ema_pack.get("decay", self.ema_decay))
            self.ema_track_buffers = bool(ema_pack.get("track_buffers", self.ema_track_buffers))
            self.ema = EMAModel(self.model, decay=float(self.ema_decay), track_buffers=bool(self.ema_track_buffers))

        # грузим EMA state
        self.ema.load_state_dict(ema_pack["state"])

        # ✅ "по умолчанию" применяем EMA веса сразу после загрузки
        prefer = checkpoint.get("prefer_ema_weights", self.prefer_ema_weights)
        if bool(prefer):
            self.ema.copy_to(self.model, strict=False)

    # ---------- validation ----------
    def validation_step(self, batch, batch_idx):
        if (not self.use_ema) or (self.ema is None) or (not self.use_ema_on_val):
            return self._validation_step_impl(batch, batch_idx, prefix="val", log_progbar=True)

        if self.log_raw_and_ema:
            loss_raw = self._validation_step_impl(batch, batch_idx, prefix="val_raw", log_progbar=False)
            with self.ema.swap_to_ema(self._ema_target()):
                loss_ema = self._validation_step_impl(batch, batch_idx, prefix="val_ema", log_progbar=True)
            self.log("val_loss", loss_ema, on_epoch=True, prog_bar=True)
            return loss_ema

        with self.ema.swap_to_ema(self._ema_target()):
            return self._validation_step_impl(batch, batch_idx, prefix="val", log_progbar=True)

    def _validation_step_impl(self, batch, batch_idx, *, prefix: str, log_progbar: bool):
        x, y, _ = self._split_batch(batch)
        preds = self(x)
        y_t = self._prepare_targets(y)

        loss = self.calc_loss(x, y_t, preds)

        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=log_progbar, batch_size=x.size(0))

        preds_flat = preds.view(preds.size(0), -1)
        y_flat = y_t.view(y_t.size(0), -1)

        metric_dict = self.val_metrics(preds_flat, y_flat)

        fixed = {}
        for k, v in metric_dict.items():
            kk = k
            # если метрика уже с val_ — не удваиваем
            if prefix == "val" and kk.startswith("val_"):
                fixed[kk] = v
            else:
                fixed[f"{prefix}_{kk}"] = v

        self.log_dict(fixed, on_epoch=True, prog_bar=log_progbar, batch_size=x.size(0))

        acc = MAE_error(preds, y_t)
        self.log(f"{prefix}_acc", acc, prog_bar=log_progbar, on_epoch=True, batch_size=x.size(0))
        return loss



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

