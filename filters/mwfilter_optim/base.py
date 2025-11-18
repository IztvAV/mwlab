import torch


class FastMN2toSParamCalculation:
    def __init__(self, matrix_order, fbw, Q, w_min=0, w_max=0, w_num=0, wlist=None, device='cpu'):
        self.matrix_order = matrix_order
        self.fbw = fbw  # <— сохраняем fbw
        if wlist is None:
            self.w = torch.linspace(w_min, w_max, w_num, device=device)
        else:
            self.w = torch.tensor(wlist, device=device, dtype=torch.float32)

        self.R = torch.zeros(matrix_order, matrix_order, dtype=torch.complex64, device=device)
        self.R[0, 0] = 1j
        self.R[-1, -1] = 1j

        self.I = torch.eye(matrix_order, matrix_order, dtype=torch.complex64, device=device)
        self.I[0, 0] = 0
        self.I[-1, -1] = 0

        # маска диагонали для внутренних резонаторов
        self._Gmask = torch.eye(matrix_order, matrix_order, dtype=torch.complex64, device=device)
        self._Gmask[0, 0] = 0
        self._Gmask[-1, -1] = 0

        # начальное G
        self.update_Q(torch.tensor(Q, dtype=torch.float32, device=device))

        self.w_calc = self.w.view(-1, 1, 1)

        # КЭШ ПРАВЫХ ЧАСТЕЙ: решаем сразу для [e0, eN] (две RHS)
        # (Nw, N, 2) — для RespM2 (где M без батча)
        self._rhs = torch.zeros(int(self.w.numel()), self.matrix_order, 2, dtype=torch.complex64, device=device)
        self._rhs[:, 0, 0]  = 1  # e0
        self._rhs[:, -1, 1] = 1  # eN

    def update_Q(self, Q: torch.Tensor):
        coeff = (1.0 / (self.fbw * Q)).to(torch.float32)  # real
        coeff_c = coeff.to(torch.complex64) * (1j)  # умножаем на j как на константу
        self.G = coeff_c * self._Gmask

    def RespM2(self, M, with_s22=False, PSs=None):
        """Обычная (небатчевая) реакция. PSs=(a1,a2,b1,b2) переопределяет set_phase."""
        # фазовые параметры из вызова?

        MR = M - self.R
        A = MR + self.w_calc * self.I - self.G

        X = torch.linalg.solve(A, self._rhs)  # (Nw, N, 2)
        x0 = X[:, :, 0]
        xN = X[:, :, 1]

        A00 = x0[:, 0]
        AN0 = x0[:, -1]
        ANN = xN[:, -1]

        S11 = 1 + 2j * A00
        S21 = -2j * AN0
        S22 = 1 + 2j * ANN

        if PSs is not None:
            a1, a2, b1, b2 = PSs
            a1 = torch.tensor(a1, dtype=torch.float32)
            a2 = torch.tensor(a2, dtype=torch.float32)
            b1 = torch.tensor(b1, dtype=torch.float32)
            b2 = torch.tensor(b2, dtype=torch.float32)
            phi1 = a1 + b1 * self.w
            phi2 = a2 + b2 * self.w
            f11 = torch.exp(-1j * 2.0 * phi1).to(torch.complex64)
            f22 = torch.exp(-1j * 2.0 * phi2).to(torch.complex64)
            f21 = torch.exp(-1j * (phi1 + phi2)).to(torch.complex64)
            S11 = S11 * f11
            S21 = S21 * f21
            if with_s22:
                S22 = S22 * f22

        if with_s22:
            return self.w, S11, S21, S22
        else:
            return self.w, S11, S21

    def BatchedRespM2(self, M, with_s22=False, PSs=None):
        """
        Батчевая версия. M — тензор [B, N, N].
        PSs=(a1,a2,b1,b2) переопределяет set_phase.
        """
        MR = (M - self.R).unsqueeze(1)  # [B, 1, N, N]
        I = self.I.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        G = self.G.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        w = self.w_calc.unsqueeze(0)  # [1, F, 1, 1]

        A = MR + w * I - G  # [B, F, N, N]
        Ainv = torch.linalg.inv(A)  # [B, F, N, N]

        A00 = Ainv[:, :, 0, 0]
        ANN = Ainv[:, :, -1, -1]
        AN0 = Ainv[:, :, -1, 0]

        S11 = 1 + 2j * A00
        S22 = 1 + 2j * ANN
        S21 = -2j * AN0

        # фаза
        if PSs is not None:
            a1, a2, b1, b2 = PSs
            a1 = torch.tensor(float(a1), dtype=torch.float32)
            a2 = torch.tensor(float(a2), dtype=torch.float32)
            b1 = torch.tensor(float(b1), dtype=torch.float32)
            b2 = torch.tensor(float(b2), dtype=torch.float32)
            phi1 = a1 + b1 * self.w  # [F]
            phi2 = a2 + b2 * self.w
            f11 = torch.exp(-1j * 2.0 * phi1).to(torch.complex64).unsqueeze(0)  # [1,F]
            f22 = torch.exp(-1j * 2.0 * phi2).to(torch.complex64).unsqueeze(0)
            f21 = torch.exp(-1j * (phi1 + phi2)).to(torch.complex64).unsqueeze(0)
            S11 = S11 * f11  # [B,F] * [1,F] — корректно бродкастится
            S21 = S21 * f21
            if with_s22:
                S22 = S22 * f22

        if with_s22:
            return self.w, S11, S21, S22
        else:
            return self.w, S11, S21
