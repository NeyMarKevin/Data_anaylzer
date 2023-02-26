import numpy as np
import Cutting
# this script is for pre-processing the FFT from the logger after cutting


def nyquist_pre_process(file_input, channel):
    features = []
    if type(file_input) is str:
        for i in channel:
            features.append(Cutting.pre_processing(file_input, i))
    else:
        for i in channel:
            features.append(file_input[:i - 1])
    return features


def get_current(data, Shunt):   # this could be changed when the channel is changed, now the first channel is measuring the shunt vol
    row = len(data[0])
    current = []
    for i in range(row):
        current.append(data[0][i] / Shunt) # Shunt is another changeable value, it depends on which shunt are wu using
    return current


def get_impedance(data, Shunt): # this will get impedance in time domain by R = U/I
    current = get_current(data, Shunt)
    voltage = data[1]
    impedance = []
    for i in range(len(current)):
        if current[i] != 0:
            R = voltage[i]/current[i]
            impedance.append(R)
        else:
            continue
    return impedance


def get_voltage(file):  # get voltage, can be changed
    voltage = file[0]
    return voltage


def main_impedance_t_domain(file, Shunt):
    real_time_impedance = get_impedance(file, Shunt)
    return real_time_impedance


def fitting_func(file): # a fucntion which can fitting the curve, still in progress....
    x = np.linspace(0, 0.001 * len(file), num=len(file))
    y = file


def get_col(data, num):
    col = []
    for each in data:
        col.append(each[num])
    return col


def get_current_logger(data, Shunt):   # this could be changed when the channel is changed, now the first channel is measuring the shunt vol
    row = len(data)
    current = []
    for i in range(row):
        current.append(data[i] / Shunt) # Shunt is another changeable value, it depends on which shunt are wu using
    return current
