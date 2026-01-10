import numpy as np
import soundfile as sf

## TP4 
from QuantCod import Fuquant
from FourierCT import TFCT

PATH = "Sons/suzanneVega_tomsDiner.wav"

x_vect, fs = sf.read(PATH)
Nwin = 2048
Nfft = 2048
Nhop = 1024
x_mat,t_vect,freq_vect = TFCT(Nwin, Nhop, Nfft, x_vect, fs)

D = 392000
L = int((fs-Nwin)/Nhop + 1)
K1 = int((Nwin-1)/Nhop)
K2 = int((fs-1)/Nhop)-L+1
N_avant = K1 - (Nhop*K1*(K1+1))/(2*Nwin)
N_apres = (K2*(fs - (L - 1)*Nhop)/Nwin) - (Nhop*K2*(K2 + 1)/(2*Nwin))
Ntr = N_avant + L + N_apres
Nb_max = int(D/Ntr)

x_mat_abs_norm = np.zeros(x_mat.shape)
A = np.zeros(x_mat.shape[0])

for n in range(x_mat.shape[1]):
    abs_mat = np.abs(x_mat[:, n])
    if max(abs_mat) != 0:
        A[n] = 1 / max(abs_mat)
        x_mat_abs_norm[:, n] = A[n] * abs_mat


x_mat_dB = np.zeros(x_mat_abs_norm.shape)
Q = np.zeros(x_mat_abs_norm.shape, dtype=int)


epsilon = 1e-10
x_mat_dB = 20 * np.log10(np.maximum(x_mat_abs_norm, epsilon))

masq = -96 

for n in range(x_mat.shape[1]):
    bits = Nb_max
    m = np.argmax(x_mat_dB[:, n])
    while x_mat_dB[m, n] > masq and bits > 0:
        x_mat_dB[m, n] -= 6
        Q[m, n] += 1
        bits -= 1
        m = np.argmax(x_mat_dB[:, n])

xq = np.zeros(x_mat.shape)

for n in range(x_mat.shape[1]):
    for k in range(x_mat.shape[0]): 
        xq[k,n] = Fuquant(x_mat_abs_norm[k,n], Q[k,n])


def TP4(D,  masq = -96):
    PATH = "Sons/suzanneVega_tomsDiner.wav"
    x_vect, fs = sf.read(PATH)
    Nwin = 2048
    Nfft = 2048
    Nhop = 1024
    x_mat,t_vect,freq_vect = TFCT(Nwin, Nhop, Nfft, x_vect, fs)

    L = int((fs-Nwin)/Nhop + 1)
    K1 = int((Nwin-1)/Nhop)
    K2 = int((fs-1)/Nhop)-L+1
    N_avant = K1 - (Nhop*K1*(K1+1))/(2*Nwin)
    N_apres = (K2*(fs - (L - 1)*Nhop)/Nwin) - (Nhop*K2*(K2 + 1)/(2*Nwin))
    Ntr = N_avant + L + N_apres
    Nb_max = int(D/Ntr)

    x_mat_abs_norm = np.zeros(x_mat.shape)
    A = np.zeros(x_mat.shape[0])

    for n in range(x_mat.shape[1]):
        abs_mat = np.abs(x_mat[:, n])
        if max(abs_mat) != 0:
            A[n] = 1 / max(abs_mat)
            x_mat_abs_norm[:, n] = A[n] * abs_mat


    x_mat_dB = np.zeros(x_mat_abs_norm.shape)
    Q = np.zeros(x_mat_abs_norm.shape, dtype=int)


    epsilon = 1e-10
    x_mat_dB = 20 * np.log10(np.maximum(x_mat_abs_norm, epsilon))


    for n in range(x_mat.shape[1]):
        bits = Nb_max
        m = np.argmax(x_mat_dB[:, n])
        while x_mat_dB[m, n] > masq and bits > 0:
            x_mat_dB[m, n] -= 6
            Q[m, n] += 1
            bits -= 1
            m = np.argmax(x_mat_dB[:, n])

    xq = np.zeros(x_mat.shape)

    for n in range(x_mat.shape[1]):
        for k in range(x_mat.shape[0]): 
            xq[k,n] = Fuquant(x_mat_abs_norm[k,n], Q[k,n])

    return xq, Q, A