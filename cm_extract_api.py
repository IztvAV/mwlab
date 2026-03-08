from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple, List, Protocol

import numpy as np
import torch
from torch import nn
import skrf as rf

import configs as cfg
import common
from losses import CustomLosses

from mwlab import TouchstoneData
from mwlab.transforms.s_transforms import S_Resample

from filters import MWFilter, CouplingMatrix
from filters.codecs import MWFilterTouchstoneCodec
from filters.mwfilter_optim.base import FastMN2toSParamCalculation
from filters.mwfilter_optim.bfgs import optimize_cm
import phase

PathLike = Union[str, Path]
PhaseInit = Tuple[float, float, float, float]  # (a11, a22, b11, b22)


# =============================================================================
# IO Spec
# =============================================================================
@dataclass(frozen=True)
class ModelIOSpec:
    # вход сети
    y_channels: List[str]
    # выход сети (порядок важен!)
    x_keys: List[str]

    # архитектура/конфиг (если нужно)
    model_name: str = "resnet_with_correction"
    model_cfg: Optional[Dict[str, Any]] = None

    def resolved_model_cfg(self) -> Dict[str, Any]:
        if self.model_cfg is not None:
            return dict(self.model_cfg)
        return {"in_channels": len(self.y_channels), "out_channels": len(self.x_keys)}


# =============================================================================
# Calibration state
# =============================================================================
@dataclass
class CalibrationState:
    n_required: int = 5
    count: int = 0
    arr: List[Dict[str, float]] = field(default_factory=list)
    calibrated: bool = False
    res: Optional[Dict[str, float]] = None

    def push(self, item: Dict[str, float]) -> None:
        self.arr.append(item)
        self.count += 1

    def try_finalize(self) -> bool:
        if self.calibrated:
            return True
        if self.count < self.n_required:
            return False

        keys = ['phi1_c', 'phi2_c', 'b11_opt', 'b22_opt']
        avg = {k: 0.0 for k in keys}
        for k in keys:
            avg[k] = float(sum(e[k] for e in self.arr) / self.count)
        self.res = avg
        self.calibrated = True
        return True


# =============================================================================
# Correction interfaces + contexts (Variant A)
# =============================================================================
class CorrectionMethod(Protocol):
    name: str

    def is_compatible(self, api: "MWFilterAPI") -> Tuple[bool, str]:
        ...

    def correct(
        self,
        measured: rf.Network,
        pred_filter: MWFilter,
        api: "MWFilterAPI",
        *,
        phase_init: Optional[PhaseInit] = None,
    ) -> MWFilter:
        ...


@dataclass(frozen=True)
class OptimCorrectionContext:
    work_model: Any
    phase_extractor: Any


@dataclass(frozen=True)
class OnlineCorrectionContext:
    work_model: Any
    phase_extractor: Any
    fast_calc: FastMN2toSParamCalculation
    inference_model_ft: Any
    corr_optim: torch.optim.Optimizer


class OptimizerCMCorrection:
    """
    Коррекция предсказаний CM через optimize_cm.
    phase_init вычисляется в MWFilterAPI.predict_cm() и сюда передаётся.
    """
    name = "optim"

    def is_compatible(self, api: "MWFilterAPI") -> Tuple[bool, str]:
        try:
            api.get_optim_correction_context()
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def correct(
        self,
        measured: rf.Network,
        pred_filter: MWFilter,
        api: "MWFilterAPI",
        *,
        phase_init: Optional[PhaseInit] = None,
    ) -> MWFilter:
        ctx = api.get_optim_correction_context()
        w = ctx.work_model.orig_filter.f_norm

        optim_filter, phase_opt = optimize_cm(
            pred_filter,
            measured,
            phase_init=phase_init,
            plot=False
        )

        optim_filter = ctx.phase_extractor.remove_phase_from_coeffs(optim_filter, w, *phase_opt)
        return optim_filter


class OnlineCMCorrection:
    """
    Онлайн-коррекция (градиентный шаг по части параметров inference_model_ft).
    Требует наличие выходов: m_*, Q, a11,a22,b11,b22.

    phase_init сюда может приходить, но не используется (оставлено для единообразия сигнатуры).
    """
    name = "online"

    def __init__(self, steps: int = 50):
        self.steps = int(steps)

    def is_compatible(self, api: "MWFilterAPI") -> Tuple[bool, str]:
        # 1) проверка готовности контекста
        try:
            api.get_online_correction_context()
        except Exception as e:
            return False, str(e)

        # 2) проверка состава выходов
        required = {"Q", "a11", "a22", "b11", "b22"}
        xk = set(api.io.x_keys)
        miss = sorted(required - xk)
        if miss:
            return False, f"model output missing keys: {miss}"
        if not any(k.startswith("m_") for k in api.io.x_keys):
            return False, "model output has no m_* keys"
        return True, "ok"

    @staticmethod
    def _factors_from_preds(preds: Dict[str, Any], links: List[Tuple[int, int]], ref_tensor: torch.Tensor) -> torch.Tensor:
        vals = []
        for (i, j) in links:
            k1 = f"m_{i}_{j}"
            k2 = f"m_{j}_{i}"
            v = preds.get(k1, preds.get(k2, None))
            if v is None:
                v = torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)
            if not torch.is_tensor(v):
                v = torch.tensor(v, device=ref_tensor.device, dtype=ref_tensor.dtype)
            vals.append(v)
        return torch.stack(vals, dim=0)

    def correct(
        self,
        measured: rf.Network,
        pred_filter: MWFilter,
        api: "MWFilterAPI",
        *,
        phase_init: Optional[PhaseInit] = None,
    ) -> MWFilter:
        ctx = api.get_online_correction_context()

        codec = ctx.work_model.codec

        loss = nn.L1Loss(reduction="sum")

        codec_db = copy.deepcopy(codec)
        codec_db.y_channels = ['S1_1.db', 'S2_1.db']
        codec_mag = copy.deepcopy(codec)
        codec_mag.y_channels = ['S1_1.mag', 'S2_1.mag']

        # measured должен быть уже в "инференс-формате":
        # - ресемпл
        # - sign flip
        # - phase_extract калибровкой (сделано в MWFilterAPI.predict_cm)
        ts = TouchstoneData(measured)
        s = codec.encode(ts)[1].unsqueeze(0).to(ctx.inference_model_ft.device)
        s_db = codec_db.encode(ts)[1].unsqueeze(0).to(ctx.inference_model_ft.device)
        s_mag = codec_mag.encode(ts)[1].unsqueeze(0).to(ctx.inference_model_ft.device)

        w = ctx.work_model.orig_filter.f_norm.to(ctx.inference_model_ft.device)
        keys = list(api.io.x_keys)

        links = pred_filter.coupling_matrix.links
        m_key_order = [f"m_{i}_{j}" for (i, j) in links]

        for _ in range(self.steps):
            x_pred = ctx.inference_model_ft(s)
            if ctx.inference_model_ft.scaler_out is not None:
                x_pred = ctx.inference_model_ft._apply_inverse(ctx.inference_model_ft.scaler_out, x_pred)

            preds = dict(zip(keys, x_pred.squeeze(0)))

            m_factors = self._factors_from_preds(preds, links, x_pred).unsqueeze(0)
            M = CouplingMatrix.from_factors(m_factors, links, pred_filter.coupling_matrix.matrix_order)

            ctx.fast_calc.update_Q(preds["Q"])
            _, s11_pred, s21_pred, s22_pred = ctx.fast_calc.RespM2(M, with_s22=True)

            a11 = preds["a11"]
            a22 = preds["a22"]
            b11 = preds["b11"]
            b22 = preds["b22"]

            phi11 = -2 * (a11 + b22 * w)
            phi22 = -2 * (a22 + b22 * w)
            phi21 = 0.5 * (phi11 + phi22)

            s11_corr = s11_pred * torch.exp(-1j * phi11)
            s22_corr = s22_pred * torch.exp(-1j * phi22)
            s21_corr = s21_pred * torch.exp(-1j * phi21)

            s_corr = torch.stack([s11_corr, s21_corr]).unsqueeze(0)  # [B, 2, L]

            err = loss(MWFilter.to_db(s_corr), s_db)
            reg = torch.max(torch.abs(torch.abs(s_corr) - s_mag)) * 2

            ctx.corr_optim.zero_grad()
            (err + reg).backward()
            ctx.corr_optim.step()

        with torch.no_grad():
            x_pred = ctx.inference_model_ft(s)
            if ctx.inference_model_ft.scaler_out is not None:
                x_pred = ctx.inference_model_ft._apply_inverse(ctx.inference_model_ft.scaler_out, x_pred)

            preds = dict(zip(keys, x_pred.squeeze(0)))
            m_factors = self._factors_from_preds(preds, links, x_pred)

            total_pred_prms = dict(zip(m_key_order, m_factors))
            correct_pred_fil = ctx.work_model.create_filter_from_prediction(
                measured, ctx.work_model.orig_filter, total_pred_prms
            )

        return correct_pred_fil


# =============================================================================
# Main API
# =============================================================================
class MWFilterAPI:
    """
    Один экземпляр = одна модель + своё состояние.
    Коррекция CM выполняется Strategy-объектами (CorrectionMethod).
    """

    def __init__(
        self,
        manifest_path: Optional[PathLike],
        *,
        io: ModelIOSpec,
        calibration_n: int = 5,
    ):
        self.manifest_path = str(manifest_path) if manifest_path is not None else None
        self.io = io

        self.configs: Optional[cfg.Configs] = self._load_configs()
        self.work_model: Optional[common.WorkModel] = None
        self.codec: Optional[MWFilterTouchstoneCodec] = None

        self.inference_model = None
        self.inference_model_ft = None
        self.corr_optim: Optional[torch.optim.Optimizer] = None
        self.corr_fast_calc: Optional[FastMN2toSParamCalculation] = None

        self.phase_extractor: Optional[phase.PhaseLoadingExtractor] = None
        self.calib = CalibrationState(n_required=calibration_n)

    # -------------------------
    # Config / bootstrap
    # -------------------------
    def _load_configs(self) -> cfg.Configs:
        if self.manifest_path is None:
            print("Load default configs")
            return cfg.Configs.init_from_default(r"D:\Burlakov\pyprojects\mwlab\default.yml")
        print(f"Load configs from: {self.manifest_path}")
        return cfg.Configs(self.manifest_path)

    def _build_work_model_and_codec(self, *, is_inference: bool) -> None:
        self.work_model = common.WorkModel(self.configs, is_inference=is_inference)

        self.codec = MWFilterTouchstoneCodec.from_dataset(
            ds=self.work_model.ds,
            keys_for_analysis=self.io.x_keys,
        )
        self.codec.y_channels = list(self.io.y_channels)
        self.codec.x_keys = list(self.io.x_keys)

        self.work_model.setup(
            model_name=self.io.model_name,
            model_cfg=self.io.resolved_model_cfg(),
            dm_codec=self.codec
        )

    # -------------------------
    # Train / Load
    # -------------------------
    def train_model(
        self,
        *,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Any] = None,
        strategy_type: str = "standard",
    ) -> Any:
        self._build_work_model_and_codec(is_inference=False)
        assert self.work_model is not None

        optimizer_cfg = optimizer_cfg or {"name": "AdamW", "lr": 9.4e-4, "weight_decay": 1e-5}
        scheduler_cfg = scheduler_cfg or {"name": "StepLR", "step_size": 25, "gamma": 0.09}
        loss_fn = loss_fn or CustomLosses("sqrt_mse_with_l1", weight_decay=1, weights=None)

        lit_model = self.work_model.train(
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            loss_fn=loss_fn,
            strategy_type=strategy_type,
        )
        return lit_model

    def load_model(self, checkpoint_path: Optional[PathLike] = None) -> None:
        self._build_work_model_and_codec(is_inference=True)
        assert self.work_model is not None
        assert self.configs is not None

        ckpt = str(checkpoint_path) if checkpoint_path is not None else self.configs.MODEL_CHECKPOINT_PATH

        self.inference_model = self.work_model.inference(ckpt)

        # online infra: копия модели + оптимизатор
        self.inference_model_ft = copy.deepcopy(self.inference_model)
        self.inference_model_ft.eval()

        for p in self.inference_model_ft.model.parameters():
            p.requires_grad = False
        for p in self.inference_model_ft.model._main_model.fc.parameters():
            p.requires_grad = True
        for p in self.inference_model_ft.model._correction_model.parameters():
            p.requires_grad = True

        params = [p for p in self.inference_model_ft.parameters() if p.requires_grad]
        self.corr_optim = torch.optim.AdamW(params=params, lr=5.370623202982373e-4, weight_decay=1e-5)

        self.corr_fast_calc = FastMN2toSParamCalculation(
            matrix_order=self.work_model.orig_filter.coupling_matrix.matrix_order,
            wlist=self.work_model.orig_filter.f_norm,
            Q=self.work_model.orig_filter.Q,
            fbw=self.work_model.orig_filter.fbw,
            device=self.inference_model_ft.device,
        )

        self.phase_extractor = phase.PhaseLoadingExtractor(self.inference_model, self.work_model, self.work_model.orig_filter)

        self.reset_calibration()

    def reset_calibration(self) -> None:
        self.calib = CalibrationState(n_required=self.calib.n_required)

    # -------------------------
    # Context getters (Variant A)
    # -------------------------
    def get_optim_correction_context(self) -> OptimCorrectionContext:
        self._assert_calibrated()
        if self.work_model is None or self.phase_extractor is None:
            raise RuntimeError("work_model/phase_extractor not ready")
        return OptimCorrectionContext(work_model=self.work_model, phase_extractor=self.phase_extractor)

    def get_online_correction_context(self) -> OnlineCorrectionContext:
        self._assert_calibrated()
        if self.work_model is None or self.phase_extractor is None:
            raise RuntimeError("work_model/phase_extractor not ready")
        if self.corr_fast_calc is None:
            raise RuntimeError("corr_fast_calc not ready")
        if self.inference_model_ft is None or self.corr_optim is None:
            raise RuntimeError("inference_model_ft/corr_optim not ready")
        return OnlineCorrectionContext(
            work_model=self.work_model,
            phase_extractor=self.phase_extractor,
            fast_calc=self.corr_fast_calc,
            inference_model_ft=self.inference_model_ft,
            corr_optim=self.corr_optim,
        )

    # -------------------------
    # Calibration / phase
    # -------------------------
    def calibrate(self, fil: rf.Network) -> bool:
        self._assert_ready_for_inference()

        if self.calib.calibrated:
            return True

        print("Calibration")
        s_resample = S_Resample(301)
        fil = s_resample(fil)

        fil.s[:, 0, 1] *= -1
        fil.s[:, 1, 0] *= -1

        w = self.work_model.orig_filter.f_norm
        res = self.phase_extractor.extract_all(fil, w_norm=w)
        self.calib.push(res)

        return self.calib.try_finalize()

    def phase_extract(self, fil: rf.Network) -> rf.Network:
        self._assert_calibrated()

        w = self.work_model.orig_filter.f_norm
        a11 = self.calib.res['phi1_c']
        a22 = self.calib.res['phi2_c']
        b11 = self.calib.res['b11_opt']
        b22 = self.calib.res['b22_opt']
        return self.phase_extractor.remove_phase_from_coeffs(fil, w, a11=a11, b11=b11, a22=a22, b22=b22)

    # -------------------------
    # Predict CM + correction
    # -------------------------
    def predict_cm(
        self,
        fil: rf.Network,
        *,
        correction: CorrectionMethod,
        decimals: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказать CM (и др. параметры из x_keys), собрать pred_filter и выполнить коррекцию выбранным методом.
        Возвращает S и матрицу связи в исходной сетке частот входного fil.
        """
        self._assert_calibrated()

        resample_in = S_Resample(301)
        resample_out = S_Resample(len(fil.f))

        fil_in = resample_in(fil)
        fil_in.s = np.round(fil_in.s, decimals)

        fil_in.s[:, 0, 1] *= -1
        fil_in.s[:, 1, 0] *= -1

        # 1) снять фазу калибровкой
        ntw_de = self.phase_extract(fil_in)

        # 2) базовое предсказание параметров
        pred_prms = self.inference_model.predict_x(ntw_de, decimals=decimals)

        # 3) собрать фильтр
        pred_fil = self.work_model.create_filter_from_prediction(fil_in, self.work_model.orig_filter, pred_prms)

        # 4) как было: калибровка + AI поправки => phase_init и применить к pred_fil
        w = self.work_model.orig_filter.f_norm
        a11_final = self.calib.res['phi1_c'] + pred_prms.get("a11", 0.0)
        a22_final = self.calib.res['phi2_c'] + pred_prms.get("a22", 0.0)
        b11_final = pred_prms.get("b11", 0.0) + self.calib.res['b11_opt']
        b22_final = pred_prms.get("b22", 0.0) + self.calib.res['b22_opt']
        phase_init: PhaseInit = (a11_final, a22_final, b11_final, b22_final)

        pred_fil = self.phase_extractor.remove_phase_from_coeffs(pred_fil, w, a11_final, b11_final, a22_final, b22_final)

        # 5) коррекция
        ok, reason = correction.is_compatible(self)
        if not ok:
            raise RuntimeError(f"Correction '{correction.name}' incompatible: {reason}")

        corrected = correction.correct(
            measured=fil_in,
            pred_filter=pred_fil,
            api=self,
            phase_init=phase_init,
        )

        # 6) вернуть в исходную сетку частот
        fil_out = resample_out(corrected)
        return fil_out.s, fil_out.coupling_matrix.matrix.numpy()

    # -------------------------
    # Utils
    # -------------------------
    def model_info(self) -> Any:
        assert self.work_model is not None
        return self.work_model.info()

    @staticmethod
    def calc_s_params(M: np.ndarray, f0: float, bw: float, Q: float, frange: Union[list, np.ndarray]):
        fbw = bw / f0
        return MWFilter.response_from_coupling_matrix(M=M, f0=f0, FBW=fbw, Q=Q, frange=frange)

    # -------------------------
    # Guards
    # -------------------------
    def _assert_ready_for_inference(self) -> None:
        if self.work_model is None or self.inference_model is None or self.phase_extractor is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

    def _assert_calibrated(self) -> None:
        self._assert_ready_for_inference()
        if not self.calib.calibrated or self.calib.res is None:
            raise RuntimeError("Not calibrated yet. Call calibrate() until it returns True.")


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Пример: CM-модель, которая умеет m_*, Q, a11/a22/b11/b22.
    # Порядок x_keys должен соответствовать обучению модели!
    io_cm = ModelIOSpec(
        y_channels=["S1_1.real", "S2_2.real", "S1_1.imag", "S2_2.imag"],
        x_keys=[
            "m_1_2", "m_2_3", "m_3_4",
            "Q",
            "a11", "a22", "b11", "b22",
        ],
    )

    api = MWFilterAPI(
        manifest_path="D:/Burlakov/pyprojects/mwlab/cm.yml",
        io=io_cm,
        calibration_n=5,
    )
    api.load_model("D:/Burlakov/ckpts/cm.ckpt")

    # ntw: rf.Network (измерение)
    # while not api.calibrate(ntw): ...
    # S1, M1 = api.predict_cm(ntw, correction=OptimizerCMCorrection())
    # S2, M2 = api.predict_cm(ntw, correction=OnlineCMCorrection(steps=60))
    pass