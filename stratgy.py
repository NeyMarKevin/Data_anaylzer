import matplotlib.pyplot as plt
import Cutting
import FFTrans
import Nyquist_preprocessing
import scipy
import numpy as np
import curve_fitting
import signal_test
from numpy import polynomial as P

import noise_canceling_method
import read_scanning_file
# this is the package that using the other script to get the result


def get_final_V(file, channel): # this function will import the voltage data from the datalogger and try different filter or method to cancel the noise
    res = []
    for i in channel:
        if type(file) is str:
            data = Cutting.pre_processing(file, i)
        else:
            data = file
        res.append(data)
    R = Nyquist_preprocessing.get_voltage(res)
    e = curve_fitting.sin_curve_fit(R, [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2,1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.09375], 167300)
    plt.figure()
    plt.plot(e)
    plt.plot(R)
    plt.title("smooth data V")
    plt.show()
    # test_sin = signal_test.generate_sinusoid(10000, 1, 256, 1000, 0)
    #Noise = noise_canceling_method.main_noise_cancelling(file, [3,4,5,6,7,8], mode='time_domain')
    #real = list(map(lambda x: x[0]-x[1], zip(R, Noise)))
    #Noise = noise_canceling_method.main_noise_cancelling(file, [3,4,5,6,7,8], mode='frequency_domain')
    #cut_r = Cutting.main_cutting_fun(R, 'constant_cutting') # cut
    # noise_canceling_method.time_predict(R) # 指数拟合
    # med_filter = scipy.signal.medfilt(test_sin, 11)
    # #smooth_data = scipy.signal.savgol_filter(mini_data, 5, 3)   # smooth  # plt
    # smooth_data2 = noise_canceling_method.np_move_Avg(mini_data, 3)
    # filtData = noise_canceling_method.np_move_Avg(smooth_data2, 50)
    # filtData2 = noise_canceling_method.np_move_Avg(filtData, 50)
    # filtData3 = noise_canceling_method.np_move_Avg(filtData2, 50)
    # filtData4 = noise_canceling_method.np_move_Avg(filtData3, 30)
    #filtData3 = scipy.signal.savgol_filter(filtData2, 51, 3)
    #filtData_bd = Cutting.band_pass_filter(filt_data2, 0, 0.000001, 20) # band pass filter
    #final_V = FFTrans.goertzel_fft(e, sample_rate=1000, get_frequencies=[256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2,1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.09375])  # plt
    #final_V = list(map(lambda x: x[0]-x[1], zip(V, Noise)))
    #final_V = Noise
    return e

def get_final_A(file, channel): # this function will import the voltage data from the datalogger and try different filter to cancel the noise
    res = []
    for i in channel:
        if type(file) is str:
            data = Cutting.pre_processing(file, i)
        else:
            data = file
        res.append(data)
    R = Nyquist_preprocessing.get_current(res, 0.2)
    e = curve_fitting.sin_curve_fit(R, [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2,1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.09375], 167340)
    plt.figure()
    plt.plot(e)
    plt.plot(R)
    plt.title("smooth data A")
    plt.show()
    # med_filter = scipy.signal.medfilt(R, 7)
    # cut_r = Cutting.main_cutting_fun(med_filter, 'constant_cutting')
    # # filt_data1 = Cutting.low_pass_filter(cut_r, 0, 400)
    # # filt_data2 = Cutting.low_pass_filter(filt_data1, 0, 300)
    # # filt_data3 = Cutting.low_pass_filter(filt_data2, 0, 300)
    # mini_data = noise_canceling_method.main_miniwave(cut_r, 1)
    # smooth_data = scipy.signal.savgol_filter(mini_data, 501, 5)
    # filtData = noise_canceling_method.curve_fitting_method(smooth_data, FFTrans.frequency)
    # # filtData = Cutting.band_pass_filter(smooth_data, 0, 90, 102)
    # # plt.figure()
    # # plt.title("final_A after passing filter")
    # read_scanning_file.draw_figure_data(filtData)
    # final_A = FFTrans.goertzel_fft(e, sample_rate=1000, get_frequencies=[256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2,1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.09375])
    # read_scanning_file.draw_FFT_figure(final_A, FFTrans.frequency)
    return e


def get_final_V_eis(file, channel): # this function will import the voltage data from the datalogger and try different filter to cancel the noise
    res = []
    for i in channel:
        if type(file) is str:
            data = Cutting.pre_processing(file, i)
        else:
            data = file
        res.append(data)
    R = Nyquist_preprocessing.get_voltage(res)
    # R1 = scipy.signal.medfilt(R, 7)
    # R2 = scipy.signal.medfilt(R1, 7)
    # R3 = noise_canceling_method.main_miniwave(R2, 0)
    curve, origin = curve_fitting.trend_fit(R, FFTrans.frequency, 166400)
    R1 = list(map(lambda x: x[0]-x[1], zip(origin, curve)))
    x = np.arange(0, len(R1), 1)
    y = R[166400 - len(R1):166400]
    p = P.polynomial.Polynomial.fit(x, y, deg=3)
    plt.plot(x, p(x), 'r', label='Power series')
    R2 = list(map(lambda x: x[0] - x[1], zip(origin, curve)))
    R2 = scipy.signal.medfilt(R1, 7)
    R3 = scipy.signal.medfilt(R2, 11)
    R4 = noise_canceling_method.main_miniwave(R3, 0)
    R5 = noise_canceling_method.main_miniwave(R4, 0)
    plt.figure()
    plt.title('fix_curve')
    plt.plot(R1)
    plt.show()
    return R5


def get_final_A_eis(file, channel): # this function will import the voltage data from the datalogger and try different filter to cancel the noise
    res = []
    for i in channel:
        if type(file) is str:
            data = Cutting.pre_processing(file, i)
        else:
            data = file
        res.append(data)
    R = Nyquist_preprocessing.get_current(res, 0.2)
    R1 = scipy.signal.medfilt(R, 7)
    R2 = scipy.signal.medfilt(R1, 7)
    R3 = noise_canceling_method.main_miniwave(R2, 0)
    mean = np.mean(R3)
    cut = R3[:166460]
    e = 1
    plt.figure()
    plt.title('a')
    plt.plot(cut)
    plt.show()
    return e


def get_final_R(file, channel_V, channel_A):    # this function is not used anymore
    # r = []
    # f_v= list(get_final_V_eis(file, channel_V))
    # f_a= list(get_final_A_eis(file, channel_A))
    # while len(f_a) != 0 and len(f_v) != 0:
    #     a = f_a.pop()
    #     v = f_v.pop()
    #     r.append(v/a)
    # res_r, para = curve_fitting.sin_curve_fit(r, FFTrans.frequency, len(r))
    # plt.figure()
    # plt.title('R')
    # plt.plot(res_r)
    # plt.show()
    return 1, 1


def get_r_freq(data):   # this will use the method in the Spectro to get the result
    img = []
    for each in data:
        img.append(FFTrans.goertzel_fft(each, 1000, FFTrans.frequency.pop()))
    return img



def get_test_signal(value, num): # this function will generate a sample data and be tested by the function to see is everything goes well
    signal = signal_test.get_constant_value(value, num)
    f, freq = noise_canceling_method.normal_fft(signal)
    plt.plot(f, freq)