import matplotlib.pyplot as plt
import numpy as np
import csv

import Cutting
import noise_canceling_method


def generate_sinusoid(N, A, f0, f1, fs, phi1, phi2):
    '''
    N(int) : number of samples
    A(float) : amplitude
    f0(float): frequency in Hz
    fs(float): sample rate
    phi(float): initial phase

    return
    x (numpy array): sinusoid signal which lenght is M
    '''

    T = 1 / fs
    n = np.arange(N)  # [0,1,..., N-1]
    x = A * np.sin(2 * f0 * np.pi * n * T + phi1) + A * np.sin(2 * f1 * np.pi * n * T + phi2)

    return x


def get_constant_value(value ,num):
    l = num
    lis = [value] * l
    return lis


def read_unit_csv():
    path = "/Users/zhengkaiwen/Desktop/21000899_test_1_eis.csv"
    with open(path,'r', encoding='utf-8', newline='') as main_file:    # open the csv file to the main_file, include with so no need to close
            main_reader = csv.reader(main_file)                             # read the csv file
            mainData = list(main_reader)                                  # convert to list
    return np.array(mainData[1:])


class eis_Data():   # this function is not used anymore
    def __init__(self):
        self.data = read_unit_csv()
        self.voltage = self.data[:,2]
        self.current = self.data[:,3]
        self.freq = self.data[:, 4]
        self.long = len(self.voltage)


    def cutting_data(self):
        res_v = []
        res_a = []
        current = []
        voltage = []
        init_freq = []
        for i in range(self.long):
            if self.freq[i] in init_freq:
                voltage.append(float(self.voltage[i]))
                current.append(float(self.current[i]) * 1000)
            else:
                current = [float(self.current[i]) * 1000]
                voltage = [float(self.voltage[i])]
                res_v.append(voltage)
                res_a.append(current)
                init_freq.append(self.freq[i])
        return res_a, res_v, init_freq


    def curve_fit(self):    # try another curve_fit method
        a, v, f = eis_Data().cutting_data()
        A_fit = []
        A_para = []
        V_para = []
        V_fit = []
        # for i in range(len(a)):
        #     afit, apara = noise_canceling_method.curve_fitting_method(a[i], float(f[i]))
        #     vfit, vpara = noise_canceling_method.curve_fitting_method(v[i], float(f[i]))
        #     A_fit.extend(afit)
        #     A_para.append(apara)
        #     V_fit.extend(vfit)
        #     V_para.append(vpara)
        # return A_fit, V_fit, A_para, V_para
        for each in a:
            A_fit.extend(each)
        Ares = Cutting.low_pass_filter(A_fit, 0, 256000)
        plt.figure()
        plt.plot(Ares)
        plt.plot(A_fit)
        plt.show()