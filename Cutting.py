from scipy import signal
import csv
from numpy import *
# this package is a pre-processing package, cut-> filter-> smooth


def pre_processing(files_path, pre_row):  # solving the noise problem, add other 7 channels volume and making average then add to the main
    with open(files_path,'r', encoding='utf-8', newline='') as main_file:    # open the csv file to the main_file, include with so no need to close
            main_reader = csv.reader(main_file)                             # read the csv file
            mainData = list(main_reader)
    row = len(mainData)
    vol = list()
    for i in range(28, row):
        vol.append(abs(float(mainData[i][pre_row+3])))
    return vol
# important notice: the pre_row should be settled from 1


def low_pass_filter(file, pre_row, highest_frequency):  # this is a low pass filter, can handle different data type (list)\(path)
    if type(file) is str:
        data = pre_processing(file, pre_row)
    else:
        data = file                                     # reading data function-> end
    w_n = 2*highest_frequency/1000                      # setting the cutting f, the 1000 is the sample f
    b, a = signal.butter(8, w_n, 'lowpass')             # this is the low pass filter settings, we use 8-order filter
    filtedData = signal.filtfilt(b, a, data)            # Using the filter to filt the data
    return filtedData


def cutting(file, start_pos, end_pos):                  # basic cutting fuc can cut the data [start_pos, end_pos] and reassamble the rest
    data = file
    cut_res = list(data[0:start_pos])
    cut_res.extend(list(data[end_pos:-1]))
    return cut_res


def get_cut_pos(data): # this fuc is not completed, it still needs to be optimized, will cut the data without changing, and the data value always close to the avg
    avg = mean(data)
    start_pos = 0
    pos = []
    while start_pos + 2 < len(data):
        s_e = []
        init = float(data[start_pos])
        for i in range(start_pos, len(data)-1):
            if abs(float(data[i+1]) - init) <= 0.002 or avg - 0.01 < data[i+1] < avg + 0.01:
                init = data[i+1]
                if i == len(data)-2:
                    start_pos = len(data)
            else:
                if i+1 - start_pos > 500:
                    s_e.append(start_pos)
                    s_e.append(i+1)
                    start_pos = i+1
                    pos.append(s_e)
                    break
                else:
                    start_pos += 1

                    break
    print(pos)
    return pos


def main_cutting_fun(file, mode):   # this is the main cutting file, has two mode, the funtion cutting is not compeleted, the constant cutting is manual, the value needs read from the graph
    if type(file) is str:
        data = pre_processing(file, 1)
    else:
        data = file
    if mode == 'function_cutting':
        cut_pos = get_cut_pos(data)
        for i in cut_pos:
            res = cutting(data, int(i[0]), int(i[1]))
            data = res
        return res
    if mode == 'constant_cutting':
        res = data[46000:140000] # set the value by the graph
        return res


def band_pass_filter(file, pre_row, low_freq, high_freq, samplef):  # A band_pass filter
    wn = 2 * low_freq / samplef
    wn2 = 2 * high_freq / samplef
    if type(file) is str:
        data = pre_processing(file, pre_row)
    else:
        data = file
    b, a  = signal.butter(5, [wn, wn2], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData



