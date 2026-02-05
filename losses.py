import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from filters import CouplingMatrix, MWFilter
from common import WorkModel


class CustomLosses(nn.Module):
    def __init__(self, loss_name="mse", **kwargs):
        super().__init__()
        self.loss_name = loss_name.lower()
        self.params = kwargs
        self.r2 = torchmetrics.R2Score()

    def forward(self, y_pred, y_true):
        if self.loss_name == "mse":
            return F.mse_loss(y_pred, y_true)
        elif self.loss_name == "reg_with_s_parameters":
            work_model: WorkModel = self.params["work_model"]
            scaler_in = work_model.dm.scaler_in
            scaler_out = work_model.dm.scaler_out
            fast_calc = self.params["fast_calc"]
            pred_prms = work_model.codec.decode_x(scaler_out.inverse(y_pred))
            orig_prms = work_model.codec.decode_x(scaler_out.inverse(y_true))

            matrix_keys = [f"m_{r}_{c}" for r, c in work_model.orig_filter.coupling_matrix.links]
            m_true = orig_prms[matrix_keys]
            m_pred = orig_prms[matrix_keys]

            _, s11_pred, s21_pred = fast_calc.BatchedRespM2(m_pred)
            _, s11_true, s21_true = fast_calc.BatchedRespM2(m_true)
            loss = F.mse_loss(y_pred, y_true) + F.l1_loss(y_pred, y_true) + self.params["weight_decay"]*(F.l1_loss(s11_true.real, s11_pred.real) + F.l1_loss(s11_true.imag, s11_pred.imag)
                                                                             + F.l1_loss(s21_true.real, s21_pred.real) + F.l1_loss(s21_true.imag, s21_pred.imag))
            return loss
        elif self.loss_name == "mse_with_l1":
            if self.params.get("weights") is None:
                weights = 1
            else:
                weights = self.params.get("weights").to("cuda")
            y_true *= weights
            y_pred *= weights
            return F.mse_loss(y_pred, y_true) + self.params.get("weight_decay", 1)*F.l1_loss(y_pred, y_true)
        elif self.loss_name == "sqrt_mse_with_l1":
            if self.params.get("weights") is None:
                weights = 1
            else:
                weights = self.params.get("weights").to(device=y_pred.device, dtype=y_pred.dtype)
            y_true_w = y_true * weights
            y_pred_w = y_pred * weights
            return torch.sqrt(
                F.mse_loss(y_pred_w, y_true_w) + self.params.get("weight_decay", 1) * F.l1_loss(y_pred_w, y_true_w))
        elif self.loss_name == "mse_with_l1_with_threshold":
            epsilon = self.params.get("abs_error_threshold", 1e-3)  # точность до знака после запятой
            weight_decay = self.params.get("weight_decay", 1.0)

            error = torch.abs(y_pred - y_true)

            # Маска: считаем только те ошибки, которые превышают порог
            mask = (error > epsilon).float()

            # Обнуляем ошибку в пределах dead zone
            clipped_error = error * mask

            # Применяем веса, если заданы
            if self.params.get("weights") is None:
                weights = 1.0
            else:
                weights = self.params.get("weights").to("cuda")

            final_weights = weights * mask  # Маска и вес вместе

            # Используем обрезанную ошибку в MSE и L1
            mse = (clipped_error ** 2 * final_weights).sum()
            l1 = (clipped_error * final_weights).sum()

            denom = final_weights.sum()
            denom = torch.clamp(denom, min=1.0)

            return (mse + weight_decay * l1) / denom
        elif self.loss_name == "l1_with_mse":
            return self.params["weight_decay"] * F.mse_loss(y_pred, y_true) + F.l1_loss(y_pred, y_true)
        elif self.loss_name == "log_mse":
            return torch.log10(F.mse_loss(y_pred, y_true))
        elif self.loss_name == "pow_mse":
            return 10**(F.mse_loss(y_pred, y_true))
        elif self.loss_name == "rmse_l1":
            return torch.sqrt(F.mse_loss(y_pred, y_true))*F.l1_loss(y_pred, y_true)
        elif self.loss_name == "mse_rmse":
            return F.mse_loss(y_pred, y_true)*torch.sqrt(F.mse_loss(y_pred, y_true))
        elif self.loss_name == "inverse_huber":
            diff = y_pred - y_true
            abs_diff = torch.abs(diff)
            mask = abs_diff < self.params.get("delta", 1)

            l1_part = abs_diff
            l2_part = diff ** 2
            return torch.where(mask, l1_part, l2_part).mean()
        elif self.loss_name == "rmse":
            return torch.sqrt(F.mse_loss(y_pred, y_true))
        elif self.loss_name == "r2":
            return 1 - self.r2(y_pred, y_true)
        elif self.loss_name == "quad_mse":
            return F.mse_loss(y_pred, y_true)**2
        elif self.loss_name == "mae":
            return F.l1_loss(y_pred, y_true)
        elif self.loss_name == "weighted_mse":
            weights = self.params.get("weights", torch.ones_like(y_true))
            return torch.mean(weights * (y_pred - y_true) ** 2)
        elif self.loss_name == "log_cosh":
            diff = y_pred - y_true
            return torch.mean(torch.log(torch.cosh(diff + 1e-12)))
        elif self.loss_name == "tukey":
            c = self.params.get("c", 4.685)
            diff = y_pred - y_true
            u = diff / c
            mask = torch.abs(u) <= 1
            loss = torch.zeros_like(diff)
            loss[mask] = c**2 / 6 * (1 - (1 - u[mask]**2)**3)
            loss[~mask] = c**2 / 6
            return torch.mean(loss)
        elif self.loss_name == "quantile":
            quantile = self.params.get("quantile", 0.9)
            diff = y_true - y_pred
            return torch.mean(torch.max(quantile * diff, (quantile - 1) * diff))
        elif self.loss_name == "pseudo_huber":
            delta = self.params.get("delta", 1.0)
            diff = y_pred - y_true
            return torch.mean(delta**2 * (torch.sqrt(1 + (diff / delta)**2) - 1))
        elif self.loss_name == "cosine_similarity":
            y_pred_norm = F.normalize(y_pred, dim=-1)
            y_true_norm = F.normalize(y_true, dim=-1)
            cos_sim = torch.sum(y_pred_norm * y_true_norm, dim=-1)
            return torch.mean(1 - cos_sim)
        elif self.loss_name == "correlation":
            vx = y_pred - torch.mean(y_pred)
            vy = y_true - torch.mean(y_true)
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
            return 1 - corr
        elif self.loss_name == "wasserstein":
            y_pred_sorted, _ = torch.sort(y_pred.view(-1))
            y_true_sorted, _ = torch.sort(y_true.view(-1))
            return torch.mean(torch.abs(y_pred_sorted - y_true_sorted))
        elif self.loss_name == "error":
            return torch.abs((y_true - y_pred)/y_true).mean()
        else:
            raise ValueError(f"Unknown loss name: {self.loss_name}")


# Пример использования
if __name__ == "__main__":
    y_pred = torch.randn(32, 37)
    y_true = torch.randn(32, 37)

    loss_fn = CustomLosses(loss_name="pseudo_huber", delta=1.0)
    loss = loss_fn(y_pred, y_true)
    print("Loss:", loss.item())
