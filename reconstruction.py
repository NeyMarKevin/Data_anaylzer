import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy
import pandas as pd

import Cutting
import FFTrans
import Nyquist_preprocessing
import noise_canceling_method
# This script can cut the signal by frequency, and cancel the noise.


eis_dic = {
2048.0: 72,
1536.0: 96,
1024.0: 144,
768.0: 192,
512.0: 256,
384.0: 256,
256.0: 256,
192.0: 256,
128.0: 256,
96.0: 256,
64.0: 256,
48.0: 256,
32.0: 256,
24.0: 256,
16.0: 256,
12.0: 256,
8.0: 256,
6.0: 256,
4.0: 256,
3.0: 256,
2.0: 256,
1.5: 256,
1.0: 256,
0.75: 256,
0.5: 256,
0.375: 256,
0.25: 256,
0.1875: 256,
0.125: 256,
0.09375: 256}   # the dic storage the eis testing message: label: frequency, dat： number of point

class logger_data():   # This is the method to treat logger data.

    def __init__(self):
        self.path = "/Users/zhengkaiwen/Desktop/10_4_scanning1.csv" # logger file path
        self.V = Cutting.pre_processing(self.path, 2)   # Get the true voltage from the file
        self.Va = Cutting.pre_processing(self.path, 1)  # Get the current voltage from the file
        self.A = Nyquist_preprocessing.get_current_logger(self.Va, 0.2) # Calculate the current, V/R, R is the impedence of the shunt
        self.f = FFTrans.frequency  # A list which has all the frequency which Spectro generates.

    def calculate_long(self):   # This function can calculate the amount of the point in each frequency and output as a list

        long = []
        for each in self.f:
            long.append(int(4000/each) + 1)
        plt.figure()
        plt.plot(self.A)
        plt.title('data logger figure' +'Hz')
        plt.show()
        return long # from the highest freq to the lowest freq, long: [high freq, low freq]

    def split(self, ed_pos, switch):    # switch can choose cutting current or voltage
        f = FFTrans.frequency.copy()    # use f for stack, do a copy method to make sure that stack method will not impact the original data
        f_long = self.calculate_long()
        res = {}    # result is a dic with label frequency
        if switch == 'A':
            data = self.A
        elif switch == 'V':
            data = self.V
        for i in range(len(f_long)):
            start_pos = ed_pos - f_long.pop()   # ed_pos is the pos where the whole test is done and jump to the rest period
            res[f.pop()] = data[start_pos:ed_pos]   # cutting
            ed_pos = start_pos
        return res

    def draw_fig(self, freq, ed_pos, switch):   # draw the figure buy ture time in the chosen frequency
        rec = self.split(ed_pos, switch)
        points = rec[freq]
        length = len(points)
        ed = 0.001 * length     # sampling time is 0.001s, so the total amount should be mutiple by 0.001
        t = np.linspace(0, ed, length)  # the true time of the signal in specfic frequency
        # Drawing function, can be used any time
        # plt.figure()
        # plt.plot(self.V)
        # plt.title('data logger figure'+ str(freq) +'Hz')
        # plt.show()
        return t, points

    def fft_figure(self, freq, ed_pos, switch): # This function can do the fft, but without the noise cancelling method, the fft will be broken
        X, res = self.draw_fig(freq, ed_pos, switch)
        if switch == 'A':
            mean = np.mean(res)
            normal_data = [each - mean for each in res]
            FFTrans.fft(normal_data, 'data_logger', freq)
        if switch == 'V':
            FFTrans.fft(res, 'data_logger', freq)


class eis_data():   # This class deals with the eis data, the eis data only include current and voltage measured by the Spectro

    def __init__(self):
        self.path = "/Users/zhengkaiwen/Desktop/ocv_eis/Oct17/Oct17_OCV_1.csv"  # eis file path
        with open(self.path, 'r', encoding='utf-8',
                  newline='') as main_file:  # open the csv file to the main_file, include with so no need to close
            main_reader = csv.reader(main_file)  # read the csv file
            mainData = list(main_reader)
            stength = len(mainData)
            self.V = []
            self.A = []
            for i in range(1, stength):
                self.V.append(float(mainData[i][0]))
                self.A.append(float(mainData[i][1]))
        self.f = eis_dic    # this dic has the frequency of eis, and the amount of point in each frequency
        self.ed_pos = len(self.V)

    def cut(self, switch):  # can do the cut in both voltage and current, and save the result in a dic in specific frequency
        cut_res = {}
        if switch == 'V':
            data = self.V
        elif switch == 'A':
            data = self.A
        start_ed = 0
        for key in self.f:
            cut_res[key] = data[start_ed: start_ed + self.f[key]]
            start_ed += self.f[key]
        return cut_res

    def draw_figure(self, freq, switch):    # similar with the logger's draw figure function
        res = self.cut(switch)
        point_num = eis_dic[freq]
        sample_T = 4/freq
        X = np.linspace(0, sample_T, point_num)
        Y = res[freq]
        # plt.figure()
        # plt.title('eis_Data'+ str(freq) +'Hz')
        # plt.plot(X, Y)
        # plt.show()
        return X, res[freq]

    def fft_figure(self, freq, switch):     # similar with the logger's fft_figure function
        X, res = self.draw_figure(freq, switch)
        if switch == 'A':
            mean = np.mean(res)
            normal_data = [each - mean for each in res]
            FFTrans.fft(normal_data, 'eis', freq)
        if switch == 'V':
            FFTrans.fft(res, 'eis', freq)


def band(data, freq, switch):   # This is a band pass filter for canceling the noise (but not use)
    low_freq = freq - 0.005
    high_freq = 400
    if switch == 'logger':
        x = Cutting.band_pass_filter(data, 0, low_freq, high_freq, 1000)
    if switch == 'eis':
        x = Cutting.band_pass_filter(data, 0, low_freq, high_freq, freq * eis_dic[freq] / 4)
    return x


class noise_cancelling():   # The method can cancel most of the drift freq by freq, and get a ideal sin wave in the end

    def __init__(self):
        self.ed = 162513    # The ed_pos is depends on the logger's figure, need to print it first and then find the end position manually

    def create_line(self, x, k, b):     # using linear method to fit the drift， this will generate a linear function
        return k * x + b

    def curve_fit(self, freq, switch, switch2):
        if switch == 'eis':
            x, y1 = eis_data().draw_figure(freq, switch2)
        elif switch == 'data_logger':
            x, y1 = logger_data().draw_fig(freq, self.ed, switch2)
        else:
            print('switch not recognized: ', switch)
        y0 = y1[0]      # init the parameter of the linear function
        ye = y1[-1]
        xl = x[-1] - x[0]
        k0 = (ye - y0)/xl
        b0 = y1[0] - k0 * x[0]
        p = [k0, b0]
        para, _ = scipy.optimize.curve_fit(self.create_line, x, y1, p0=p, maxfev=10000) # the method in the package which can minimize the error of the fitting
        y_fit = []
        for i in range(len(x)):
            y_fit.append(y1[i] - self.create_line(x[i], *para))     # using original wave to minus this to cancel the drift
        # plt.figure()
        # plt.title(switch + 'fitting_line')
        # plt.plot(x, y_fit)
        # plt.show()
        return x, y_fit

    def fft_figure(self, freq, switch, switch2):        # Function not completed, users can see the fft figure to make sure drift is cancelled
        x, res = self.curve_fit(freq, switch, switch2)
        FFTrans.fft(res, switch, freq)


class get_impedance():  # This is the function which can calculate the impedance from the EIS data and logger data
    def __init__(self):
        self.f = 0.09375
        ns_can = noise_cancelling()
        self.eis_time, self.eis_voltage = ns_can.curve_fit(self.f, "eis", 'V')  # clear the drift from both eis and data logger
        self.t_eis, self.eis_current = ns_can.curve_fit(self.f, 'eis', 'A')
        self.logger_time, self.logger_voltage = ns_can.curve_fit(self.f, "data_logger", 'V')
        self.t_logger, self.logger_current = ns_can.curve_fit(self.f, "data_logger", 'A')


    def get_R(self, switch):        # will calculate the impedance, switch is the choose of data logger and eis
        if switch == 'eis':
            v = self.eis_voltage
            c = self.eis_current
            t = self.eis_time
            f = t[-1] / len(t)  # f is a constant list, but there are some different between
        elif switch == 'data_logger':
            v = self.logger_voltage
            c = self.logger_current
            t = self.logger_time
            f = 1000
        r = []
        for i in range(len(v)):
            r.append(v[i]/c[i])
        plt.figure()
        plt.title('impedance for ' + str(switch))
        plt.plot(t, r)
        plt.show()


class nyquist():    # this class will use the former two classes to calculate the Nyquist plot of the EIS
    def __init__(self):
        self.ed = 167345
        self.f = FFTrans.frequency.copy()
        self.path = "/Users/zhengkaiwen/Desktop/ocv_eis/Oct17/Oct17_EIS_1.csv"
        with open(self.path, 'r', encoding='utf-8',
                  newline='') as main_file:  # open the csv file to the main_file, include with so no need to close
            main_reader = csv.reader(main_file)  # read the csv file
            mainData = list(main_reader)
            stength = len(mainData)
            self.R = []
            self.I = []
            for i in range(1, stength):
                self.R.append(float(mainData[i][0]) * 1000) # the data is in mV and mA, *1000 to switch to A and V
                self.I.append(float(mainData[i][1]) * 1000)

    def sin_curve_fit(self, freq, switch1, switch2):    # This function is used to fix the sinusoidal wave
        x, sign = noise_cancelling().curve_fit(freq, switch1, switch2)  # use the curve fit function
        y_fit, para = noise_canceling_method.curve_fitting_method(sign, freq, 'sin_fit')    # using the function to fit
        A = para[0][0]  # get amptitude
        phase = para[0][2]  # get phase
        return A, phase

    def get_R(self, freq):  # get the impedance from eis data and logger data, the theory is in https://www.biologic.net/topics/what-is-eis/
        A_eis, A_eis_phase = self.sin_curve_fit(freq, 'eis', 'A')
        A_datalogger, A_datalogger_phase = self.sin_curve_fit(freq, 'data_logger', 'A')
        V_eis, V_eis_phase = self.sin_curve_fit(freq, 'eis', 'V')
        V_datalogger, V_datalogger_phase = self.sin_curve_fit(freq, 'data_logger', 'V')
        if np.sign(A_eis) != np.sign(V_eis):    # The phase has some problem, this function can help the phase
            A_eis_phase += np.pi
        amp_eis = abs(V_eis/A_eis) * 1000
        eis_phase = A_eis_phase - V_eis_phase
        if np.sign(A_datalogger) != np.sign(V_datalogger):
             A_datalogger_phase += np.pi
        amp_datalogger = abs(V_datalogger/A_datalogger) * 1000  # the abs of the amp have some problems in the highest frequecny, the phase angle needs another method to do that
        datalogger_phase = A_datalogger_phase - V_datalogger_phase
        real_eis = abs(amp_eis * np.cos(eis_phase))
        imag_eis = abs(amp_eis * np.sin(eis_phase))
        real_datalogger = abs(amp_datalogger * np.cos(datalogger_phase))    # Re = A * sin
        imag_datalogger = abs(amp_datalogger * np.sin(datalogger_phase))    # Im = A * cos
        eis_res = []
        eis_res.append(real_eis)
        eis_res.append(imag_eis)
        datalogger_res = []
        datalogger_res.append(real_datalogger)
        datalogger_res.append(imag_datalogger)
        return eis_res, datalogger_res  # calculate both eis_res and datalogger_res

    def getR_in_all_f(self):    # use the get_R function repeatly for all the freq in test
        eis_res = []
        datalogger_res = []
        dx = []
        dy = []
        x = []
        y = []
        for i in range(len(self.f)):
            eis, datalogger = self.get_R(self.f.pop())
            eis_res.append(eis)
            datalogger_res.append(datalogger)
        for each in eis_res:
            x.append(each[0])   # real value for the Spectro
            y.append(each[1])
        for each in datalogger_res:
            dx.append(each[0])  # real value for the datalogger
            dy.append(each[1])
        plt.figure()
        plt.plot(x, y, 'o')
        for i in range(5):
            plt.annotate(FFTrans.frequency[-1-i], (dx[i], dy[i]))
            plt.plot(dx[i], dy[i], 'o')
        plt.plot(self.R, self.I)
        plt.xlabel('real')
        plt.ylabel('imag')
        plt.title('Nyquist eis')
        plt.show()