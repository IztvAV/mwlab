import torch


class FastMN2toSParamCalculation:
    def __init__(self, matrix_order, fbw, Q, w_min=0, w_max=0, w_num=0, wlist=None):
        self.matrix_order = matrix_order
        if wlist is None:
            self.w = torch.linspace(w_min, w_max, w_num)
        else:
            self.w = torch.tensor(wlist)
        self.R = torch.zeros(matrix_order, matrix_order, dtype=torch.complex64)
        self.R[0, 0] = 1j
        self.R[-1, -1] = 1j
        self.I = torch.eye(matrix_order, matrix_order, dtype=torch.complex64)
        self.I[0, 0] = 0
        self.I[-1, -1] = 0
        self.G = 1j*1 * torch.eye(matrix_order, matrix_order, dtype=torch.complex64)
        self.G[0, 0] = 0
        self.G[-1, -1] = 0
        for res in range(1, matrix_order - 1):
            self.G[res, res] = 1j / (fbw * Q)
        self.S11 = torch.zeros(w_num, dtype=torch.complex64)
        self.S21 = torch.zeros(w_num, dtype=torch.complex64)
        self.S22 = torch.zeros(w_num, dtype=torch.complex64)
        self.w_calc = self.w.view(-1, 1, 1)

    def RespM2(self, M):
        # Батчевое создание матриц A
        MR = torch.tensor(M) - self.R
        A = MR + self.w_calc * self.I - self.G

        # start_time = time.time_ns()
        # Обратные матрицы
        # Ainv = torch.linalg.inv(A)  # (B, N, N)
        # stop_time = time.time_ns()
        # print(f"Time to calc inverse matrix by torch.linalg.inv(A) = {(stop_time - start_time)/1e3} usec")

        # start_time = time.time_ns()
        b = torch.zeros(A.shape[0], A.shape[1], 1, dtype=torch.complex128)
        b[:, 0, 0] = 1
        Ainv = torch.linalg.solve(A, b)
        # stop_time = time.time_ns()
        # print(f"Time to calc inverse matrix by torch.linalg.solve(A, b) = {(stop_time - start_time)/1e3} usec")

        # Расчет S-параметров
        A00 = Ainv[:, 0, 0]
        # ANN = Ainv[:, -1, -1]
        AN0 = Ainv[:, -1, 0]

        S11 = 1 + 2j * 1 * A00
        # S22 = 1 + 2j * Rl * ANN
        S21 = -2j * torch.sqrt(torch.tensor(1 * 1, dtype=torch.float32)) * AN0

        return self.w, S11, S21