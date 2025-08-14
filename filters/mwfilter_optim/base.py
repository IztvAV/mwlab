import torch


class FastMN2toSParamCalculation:
    def __init__(self, matrix_order, fbw, Q, w_min=0, w_max=0, w_num=0, wlist=None, device='cpu'):
        self.matrix_order = matrix_order
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

        self.G = 1j * 1 * torch.eye(matrix_order, matrix_order, dtype=torch.complex64, device=device)
        self.G[0, 0] = 0
        self.G[-1, -1] = 0
        for res in range(1, matrix_order - 1):
            self.G[res, res] = 1j / (fbw * Q)

        self.w_calc = self.w.view(-1, 1, 1)

    def RespM2(self, M, with_s22=False):
        MR = M - self.R
        A = MR + self.w_calc * self.I - self.G

        B, N, _ = A.shape
        e0 = torch.zeros(B, N, 1, dtype=torch.complex64, device=A.device)
        eN = torch.zeros(B, N, 1, dtype=torch.complex64, device=A.device)

        e0[:, 0, 0] = 1
        eN[:, -1, 0] = 1

        x0 = torch.linalg.solve(A, e0)  # A^{-1} @ e0
        xN = torch.linalg.solve(A, eN)  # A^{-1} @ eN

        A00 = x0[:, 0, 0]
        AN0 = x0[:, -1, 0]
        ANN = xN[:, -1, 0]

        S11 = 1 + 2j * A00
        S21 = -2j * AN0
        S22 = 1 + 2j * ANN

        if with_s22:
            return self.w, S11, S21, S22
        else:
            return self.w, S11, S21

    def BatchedRespM2(self, M, with_s22=False):
        # Приведение размеров
        MR = (M - self.R).unsqueeze(1)  # [64, 1, 8, 8]
        I = self.I.unsqueeze(0).unsqueeze(0)  # [1, 1, 8, 8]
        G = self.G.unsqueeze(0).unsqueeze(0)  # [1, 1, 8, 8]
        w = self.w_calc.unsqueeze(0)  # [1, 301, 1, 1]

        # Теперь размеры совпадают
        A = MR + w * I - G  # [64, 301, 8, 8]

        # Решение линейной системы
        # b = torch.zeros(A.shape[0], A.shape[1], A.shape[2], 1, dtype=torch.complex64, device=A.device)
        # b[:, :, 0, 0] = 1  # [64, 301, 8, 1]

        Ainv = torch.linalg.inv(A)  # [64, 301, 8, 1]

        A00 = Ainv[:, :, 0, 0]
        ANN = Ainv[:, :, -1, -1]
        AN0 = Ainv[:, :, -1, 0]

        S11 = 1 + 2j * A00
        S22 = 1 + 2j * ANN
        S21 = -2j * AN0

        if with_s22:
            return self.w, S11, S21, S22
        else:
            return self.w, S11, S21
