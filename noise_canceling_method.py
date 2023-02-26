import Cutting
import FFTrans
import read_scanning_file
import numpy as np
from scipy import fftpack
import scipy
from statsmodels.tsa.holtwinters import Holt
import pandas as pd

def get_noise_f(file, channel):
    if file and channel:
        res = []
        for i in channel:
            if type(file) is str:
                data = Cutting.main_cutting_fun(Cutting.pre_processing(file, i), 'constant_cutting')
            else:
                data = Cutting.main_cutting_fun(file, 'constant_cutting')
            res.append(data)
        for i in res:
            #f = FFTrans.frequency
            #data = FFTrans.goertzel_fft(i, sample_rate=1000, get_frequencies=f)
            f, data = normal_fft(i)
        return f, data
    else:
        return None


def normal_fft(X):
    Fs = 1000  # 采样频率
    T = 1 / Fs  # 采样周期，只相邻两数据点的时间间隔
    L = len(X)
    Y = fftpack.fft(X)
    p2 = np.abs(Y)  # 双侧频谱
    p3 = p2 / L
    f = np.arange(L)
    return f, p3


def normalize_fuc(data):
    normalize_data = []
    mean = np.mean(data)
    for each in data:
        normalize_data.append(each - mean)
    return mean


def avg_for_mutilist(data):
    normalize_db = []
    for each in data:
        normalize_db.append(normalize_fuc(each))
    for i in range(len(data[0])):
        mean = []
        for j in range(len(data)):
            mean.append(data[j][i])
        normalize_db.append(np.mean(mean))
    return normalize_db


def avg_for_complex(data):
    avg_Data= []
    for i in range(len(data[0])):
        mean_real = []
        mean_imag = []
        for p in range(len(data)):
            mean_real.append(data[p][i].real)
            mean_imag.append(data[p][i].imag)
        real = float(np.mean(mean_real))
        imag = float(np.mean(mean_imag))
        avg_Data.append(complex(real, imag))
    return avg_Data


def main_noise_cancelling(data, channel, mode):
    res = []
    for i in channel:
        if type(data) is str:
            d = Cutting.pre_processing(data, i)
        else:
            d = data
        res.append(d)
    if mode == 'time_domain':
        return avg_for_mutilist(res)

    if mode == 'frequency_domain':
        fft = []
        for each in res:
            fft.append(FFTrans.goertzel_fft(Cutting.main_cutting_fun(each, mode='constant_cutting'), 1000, get_frequencies=FFTrans.frequency))
        return avg_for_complex(fft)


import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt
from math import log

#sgn函数
def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(new_df):
    data = new_df
    w = pywt.Wavelet('sym8')#选择sym8小波基
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 5层小波分解

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))#固定阈值计算
    usecoeffs = []
    usecoeffs.append(ca5)  # 向列表末尾添加对象

    #软硬阈值折中的方法
    a = 0.5

    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    return recoeffs


# def sgn(num):
#     if(num > 0.0):
#         return 1.0
#     elif(num == 0.0):
#         return 0.0
#     else:
#         return -1.0

# def wavelet_noising(new_df):
#     data = new_df
#     w = pywt.Wavelet('dB10')#选择dB10小波基
#     ca3, cd3, cd2, cd1 = np.array(pywt.wavedec(data, w, level=3))  # 3层小波分解
#     type(ca3)
#     # ca3=ca3.squeeze(axis=0) #ndarray数组减维：(1，a)->(a,)
#     # cd3 = cd3.squeeze(axis=0)
#     # cd2 = cd2.squeeze(axis=0)
#     # cd1 = cd1.squeeze(axis=0)
#     length1 = len(cd1)
#     length0 = len(data)
#
#     abs_cd1 = np.abs(np.array(cd1))
#     median_cd1 = np.median(abs_cd1)
#
#     sigma = (1.0 / 0.6745) * median_cd1
#     lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))
#     usecoeffs = []
#     usecoeffs.append(ca3)
#
#     #软阈值方法
#     for k in range(length1):
#         if (abs(cd1[k]) >= lamda/np.log2(2)):
#             cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - lamda/np.log2(2))
#         else:
#             cd1[k] = 0.0
#
#     length2 = len(cd2)
#     for k in range(length2):
#         if (abs(cd2[k]) >= lamda/np.log2(3)):
#             cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - lamda/np.log2(3))
#         else:
#             cd2[k] = 0.0
#
#     length3 = len(cd3)
#     for k in range(length3):
#         if (abs(cd3[k]) >= lamda/np.log2(4)):
#             cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - lamda/np.log2(4))
#         else:
#             cd3[k] = 0.0
#
#     usecoeffs.append(cd3)
#     usecoeffs.append(cd2)
#     usecoeffs.append(cd1)
#     recoeffs = pywt.waverec(usecoeffs, w)#信号重构
#     return recoeffs


def main_miniwave(file, row):
    data = []
    if type(file) is str:
        data = Cutting.pre_processing(file, row)
    else:
        data = file
    data_denoising = wavelet_noising(data)#调用小波阈值方法去噪
    return data_denoising


def np_move_Avg(a, n, mode = 'valid'):
    return(np.convolve(a, np.ones((n,))/n, mode = mode))


def create_sinosudial_fuc(x, a1, w1, fai1, c):
    return a1 * np.sin(w1 * x + fai1) + c


def create_square_fuc(x, a, b, c):
    return a* x**2 + b*x + c


def create_cubic_fuc(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def create_exp_fuc(x, a, b, c):
    return a * np.exp(-1 *b * x) +c


def curve_fitting_method(file, freq, trend):
    if trend == 'sin_fit':
        pa = []
        fai1 = 0
        c = np.mean(file)
        a1 = max(file) - min(file)
        x = list(np.arange(0, 4/freq, float(4/(freq * len(file)))))
        while len(x) != len(file):
            x.pop()
            print(freq)
        w = 2 * np.pi * freq
        p = [a1, w, fai1, c]
        para, _ = scipy.optimize.curve_fit(create_sinosudial_fuc, x, file, p0=p, maxfev = 10000)
        pa.append(para)
        y_fit = []
        for each in x:
            y_fit.append(create_sinosudial_fuc(each, *para) - para[-1])
        # plt.figure()
        # plt.title('sinowave fitting method')
        # plt.plot(x, y_fit)
        # plt.show()
        return y_fit, pa

    elif trend == 'square_fit':
        pa = []
        x = np.arange(0, len(file), 1)
        para, _ = scipy.optimize.curve_fit(create_square_fuc, x, file, maxfev=10000)
        pa.append(para)
        y_fit = []
        for each in x:
            y_fit.append(create_square_fuc(each, *para))
        plt.figure()
        plt.title('square_fit')
        plt.plot(y_fit)
        plt.plot(file)
        plt.show()
        return y_fit, pa

    elif trend == 'exp_fit':
        pa = []
        x = np.arange(0, len(file), 1)
        para, _ = scipy.optimize.curve_fit(create_exp_fuc, x, file, maxfev=10000)
        pa.append(para)
        y_fit = []
        for each in x:
            y_fit.append(create_exp_fuc(each, *para))
        plt.figure()
        plt.title('square_fit')
        plt.plot(y_fit)
        plt.plot(file)
        plt.show()
        return y_fit, pa

    elif trend == 'cubic_fit':
        pa = []
        x = np.arange(0, len(file), 1)
        para, _ = scipy.optimize.curve_fit(create_cubic_fuc, x, file, maxfev=10000)
        pa.append(para)
        y_fit = []
        for each in x:
            y_fit.append(create_cubic_fuc(each, *para))
        plt.figure()
        plt.title('cubic_fit')
        plt.plot(y_fit)
        plt.plot(file)
        plt.show()
        return y_fit, pa


def calc_next_s(alpha, x):
    s = [0 for i in range(len(x))]
    s[0] = np.sum(x[0:3]) / float(3)
    for i in range(1, len(s)):
        s[i] = alpha*x[i] + (1-alpha)*s[i-1]
    return s


# 预测
def time_predict(x):
    alpha = 0.5
    s1 = calc_next_s(alpha, x) # 一次
    s2 = calc_next_s(alpha, s1)  # 二次
    s3 = calc_next_s(alpha, s2)    # 三次
    a3 = [(3 * s1[i] - 3 * s2[i] + s3[i]) for i in range(len(s3))]
    b3 = [((alpha / (2 * (1 - alpha) ** 2)) * (
                (6 - 5 * alpha) * s1[i] - 2 * (5 - 4 * alpha) * s2[i] + (4 - 3 * alpha) * s3[i])) for i in
          range(len(s3))]
    c3 = [(alpha ** 2 / (2 * (1 - alpha) ** 2) * (s1[i] - 2 * s2[i] + s3[i])) for i in range(len(s3))]
    pred = []
    for i in range(len(x)):
        pred.append(a3[i] + b3[i] * 1 + c3[i] * (1 ** 2))
    plt.plot(pred)
    plt.plot(x)
    plt.show()
    return pred


