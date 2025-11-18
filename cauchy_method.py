import numpy as np
import matplotlib.pyplot as plt

def bp2lp(freq, f0, BW):
    """Bandpass → Lowpass частотное преобразование."""
    return (f0 / BW) * (freq / f0 - f0 / freq)

def cauchy_vander_row(num, N):
    """Полный эквивалент MATLAB vander_row(num, N)."""
    row = np.ones(N + 1, dtype=complex)
    for i in range(1, N + 1):
        row[i] = num**i
    return np.flip(row)   # ← ВАЖНО

def extract_coeffs(freq, s11, s21, f0, bw, Q, N, nz):
    """
    Python-эквивалент MATLAB-функции:
        [a1, a2, b, eps, eps_r] = cauchy_coeff_extraction(...)
    """
    to_db = lambda x: 20 * np.log10(abs(x))
    # MATLAB: freq = freq.'
    freq = np.asarray(freq, complex).ravel()
    s11 = np.asarray(s11, complex).ravel()
    s21 = np.asarray(s21, complex).ravel()

    fbw = bw/f0
    # MATLAB: s = 1/(fbw*Q) + 1i * bp2lp(...)
    s = 1.0 / (fbw * Q) + 1j * bp2lp(freq, f0, bw)

    Ns = len(freq)

    Vn  = np.zeros((Ns, N  + 1), dtype=complex)
    Vnz = np.zeros((Ns, nz + 1), dtype=complex)

    # MATLAB-цикл
    for i in range(Ns):
        Vn[i, :]  = cauchy_vander_row(s[i],  N)
        Vnz[i, :] = cauchy_vander_row(s[i], nz)

    # MATLAB: S11_Vn = s11 .* Vn
    S11_Vn = s11[:, None] * Vn
    S21_Vn = s21[:, None] * Vn

    # Построение полной матрицы (как в MATLAB)
    Vx_top = np.hstack([Vn, np.zeros((Ns, nz+1), dtype=complex), -S11_Vn])
    Vx_bot = np.hstack([np.zeros((Ns, N+1), dtype=complex), Vnz, -S21_Vn])
    Vx = np.vstack([Vx_top, Vx_bot])

    # MATLAB: [U, S, V] = svd(Vx)
    # SciPy/Numpy: np.linalg.svd возвращает Vh = Vᴴ, поэтому транспонируем
    U, Svals, Vh = np.linalg.svd(Vx)
    V = Vh.conj().T

    # MATLAB: a1 = V(1:N+1,end)
    a1 = V[0:N+1, -1]
    a1 = a1 / a1[-1]  # нормировка как в MATLAB

    # MATLAB: a2 = V(N+2 : N+2+nz , end)
    a2 = V[N+1 : N+1+nz+1, -1]
    a2 = a2 / a2[-1]

    # MATLAB: b = V(N+1+nz+1+1 : end, end)
    b = V[N+1+nz+1 : , -1]
    b = b / b[-1]

    # MATLAB: if mod(N-nz,2)==0, a2 = a2 * 1i
    if (N - nz) % 2 == 0:
        a2 = a2 * 1j

    # Оценка S11_ и S21_
    S11_ = (Vn @ a1) / (Vn @ b)

    def matlab_max(vals):
        idx = np.argmax(np.abs(vals))
        vals_max = vals[idx]
        return vals_max

    # максимум по модулю, как в MATLAB
    norm = matlab_max(S11_)/matlab_max(s11)

    a1 /= norm
    a2 /= norm

    S21_ = (Vnz @ a2) / (Vn @ b)
    eps = matlab_max(abs(S21_)) / matlab_max(abs(s21))

    # MATLAB: if N == nz, eps_r = eps / sqrt(eps^2 - 1)
    eps_r = 1
    if N == nz:
        eps_r = eps / np.sqrt(eps**2 - 1)

    s_apx = 1.0 / (fbw * Q) + 1j * np.linspace(-4.0, 4.0, 1001)
    # s_apx = s

    s11_ext = np.polyval(a1, s_apx)/(np.polyval(b, s_apx))
    s21_ext = np.polyval(a2, s_apx)/(eps*np.polyval(b, s_apx))
    plt.figure()
    plt.plot(np.imag(s), to_db(s11), np.imag(s), to_db(s21))
    plt.plot(np.imag(s_apx), to_db(s11_ext), np.imag(s_apx), to_db(s21_ext))
    plt.title("Extracted S-parameters")
    plt.legend(["S11 Origin", "S21 Origin", "S11 Recover", "S21 Recover"])
    # plt.figure()
    # plt.title("ReIm parts of S21")
    # plt.plot(np.imag(s_apx), np.imag(s21_ext), np.imag(s), np.imag(s21))
    # plt.figure()
    # plt.title("ReIm parts of S11")
    # plt.plot(np.imag(s_apx), np.imag(s11_ext), np.imag(s), np.imag(s11))
    # plt.figure()
    # plt.title("Phase of S11")
    # plt.plot(np.imag(s_apx), np.angle(s11_ext), np.imag(s), np.angle(s11))
    # return a1, a2, b, eps, eps_r
    return np.imag(s_apx), s11_ext, s21_ext
