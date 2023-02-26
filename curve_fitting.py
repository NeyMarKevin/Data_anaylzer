import matplotlib.pyplot as plt

import Cutting
import database_transfer
import noise_canceling_method
import FFTrans
import scipy


freq = FFTrans.frequency



def sin_curve_fit(data, freq, ed):  # this is the function fitting, feed the data, the function will use some filter to cancel the noise, then use a sinosudial or linear function to fix the data
    end_pos = ed
    res_wave = []
    long = []
    res_para = []
    while len(freq) >= 1:
        f = freq.pop()
        long.append(4000/f)
    for each in long:
        start_pos = end_pos - int(each)
        wave = data[start_pos:end_pos]
        med_filter2 = scipy.signal.medfilt(wave, 7) # filter
        mini_data = noise_canceling_method.main_miniwave(med_filter2, 1)
        mini_data2 = noise_canceling_method.main_miniwave(mini_data, 1)
        curve, para= noise_canceling_method.curve_fitting_method(mini_data2, 4000/each, 'sin_fit')
        res_wave.extend(curve)
        res_para.append(para)
        end_pos = start_pos
    return res_wave, res_para


def trend_fit(data, freq, ed):  # this function is inside the sin_curve_fit function
    long = 0
    for each in freq:
        long += int(4000/each)
    d = data[ed - long: ed]
    curve, para = noise_canceling_method.curve_fitting_method(d, freq, 'cubic_fit')
    return curve, d

