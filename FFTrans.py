import numpy as np
import Nyquist_preprocessing
from scipy import fftpack
from math import pi, cos
from cmath import exp
import matplotlib.pyplot as plt
# this package will do the fft in different frequency (goertzel's algorithm)


frequency_quick_eis = [256, 192, 128, 96, 64, 48, 32, 24, 16]
frequency = [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.09375]
def goertzel_fft(samples, sample_rate, get_frequencies):    # the method from the Spectro unit, code is totally same
    if sample_rate > 0:
        window_size = len(samples)
        f_step = sample_rate / float(window_size)

        freq = get_frequencies
        k = freq / f_step

        A = 2 * pi * k / window_size
        B = 2 * cos(A)
        C = exp(-1j * A)
        D = exp(-1j * A * (window_size - 1))

        s0 = 0
        s1 = 0
        s2 = 0

        for i in range(window_size - 1):
            s0 = samples[i] + B * s1 - s2
            s2 = s1
            s1 = s0

        s0 = samples [window_size - 1] + B * s1 - s2
        y = s0 - s1 * C
        y = y * D /window_size
    return y


def fft(data, mode, freq):      # doing the quick fft and plot
    fft = np.fft.fft(data)
    fftshift = np.fft.fftshift(fft)
    amp = abs(fftshift) / len(fft)
    pha = np.angle(fftshift)
    if mode == 'data_logger':
        fre = np.fft.fftshift(np.fft.fftfreq(len(data), d=0.001))
    if mode == 'eis':
        fre = np.fft.fftshift(np.fft.fftfreq(len(data), d=4/(freq * len(data))))
    plt.figure()
    plt.plot(fre, amp)
    plt.xlabel('Frequency' + mode + str(freq))
    plt.ylabel('Amplitude' + mode +str(freq))
    plt.figure()
    plt.plot(fre, pha)
    plt.xlabel('Frequency'+ mode + str(freq))
    plt.ylabel('Phase'+ mode + str(freq))
    plt.show()
    return fft


def turn_to_complex(para):      # after fft, in each frequency, multiple the amp with the phase to get value
    res = []
    for each in para:
        point = []
        A = abs(each[0][0])
        phase = each[0][2]
        point.append(A * np.cos(phase))
        point.append(A * np.sin(phase))
        res.append(point)
    return res


def sin_para_method(a, v):      # function is not used anymore
    x = []
    y = []
    for i in range(len(a)):
        A = abs(v[i][0][0] / a[i][0][0])
        phase = v[i][0][2] - a[i][0][2]
        x.append(A * np.cos(phase))
        y.append(A * np.sin(phase))
    return x, y
