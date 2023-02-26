import csv
import matplotlib.pyplot as plt
import numpy as np
# this script is more like a plotting package, it has various of plotting figures method


skip_step = 1  # Too many datapoints here, this variable is going to pick up one point in every step.

def open_file(file):
    with open(file,'r', encoding='utf-8', newline='') as main_file:    # open the csv file to the main_file, include with so no need to close
            main_reader = csv.reader(main_file)                             # read the csv file
            mainData = list(main_reader)                                    # convert to list
    return mainData


def pre_process(files_path):  # solving the noise problem, add other 7 channels volume and making average then add to the main
    with open(files_path,'r', encoding='utf-8', newline='') as main_file:    # open the csv file to the main_file, include with so no need to close
            main_reader = csv.reader(main_file)                             # read the csv file
            mainData = list(main_reader)
    row = len(mainData)
    vol = list()
    for i in range(28, row):
        all_noise = 0
        for j in range(5,12):   # get 7 channels volume
            all_noise += float(mainData[i][j])
        noise = all_noise/7
        voltage = float(mainData[i][4])    # filter about the noise
        current = voltage               # get the current
        vol.append(current)
    return vol

def draw_figure(files_path):    # drawing the graph
    time = list()   # the volume in X
    volume = pre_process(files_path) # the all volume in Y
    p = 0

    row_len = len(volume)   # coding for pick some of the data in the .csv file
    final_vol = []
    while skip_step * p < row_len:
        time.append(p)
        final_vol.append(volume[skip_step * p])
        p += 1
    name = files_path.split("/")[-1]

    plt.figure()
    plt.title(name)         #use the file name to distinguish different data
    plt.plot(time, final_vol)
    plt.show()

def muti_pre_process(files_path, channel_num):  # to get more than one list from the csv file
    with open(files_path,'r', encoding='utf-8', newline='') as main_file:    # open the csv file to the main_file, include with so no need to close
            main_reader = csv.reader(main_file)                             # read the csv file
            mainData = list(main_reader)
    R = [2.5, 1, 0.2]
    rows = len(mainData)
    muti_num = []
    for j in range(4, 4+channel_num):
        vol = []
        for i in range(28,rows):
            vol.append(float(mainData[i][j])/R[j-4])
        muti_num.append(vol)
    return muti_num         # it will return a muti-dimension list

def draw_muti_figure(files_path, channel_num):              # drawing two or more graph
    volume = muti_pre_process(files_path,channel_num)       # the all volume in Y
    colume_len = len(volume[0])                             # coding for pick some of the data in the .csv file
    for i in range(0,3):
        p = 0
        time = []
        final_vol = []
        while skip_step * p < colume_len:
            time.append(p)
            final_vol.append(volume[i][skip_step * p])
            p += 1
        print(len(final_vol))
        name = files_path.split("/")[-1] + "channel" + str(i+1) # return the channel name to the figure to distinguish
        plt.figure()
        plt.title(name)                                     #use the file name to distinguish different data
        plt.plot(time, final_vol)
        plt.show()

def sum_compare(files_path,channel_num, sum_row, compare_row): # this fuction will sum two rows in the data and compare with another row
    volume = muti_pre_process(files_path,channel_num)
    summary_row = []
    column_len = len(volume[1])
    p = 0
    while p < column_len:
        summary_result = 0
        for each in sum_row:
            summary_result += float(volume[each-1][p])
            summary_row.append(summary_result)  # return the summary row
        p += 1
    compare_data = []
    summary_data = []

    q = 0
    time = []
    while skip_step * q < column_len:
        summary_data.append(summary_row[skip_step * q])
        compare_data.append(volume[compare_row-1][skip_step * q])
        time.append(q)
        q += 1
    name = 'Sum up channel {} with compare channel {}'.format(sum_row,compare_row)  # plt them in a same figure
    plt.figure()
    plt.title(name)
    plt.plot(time,compare_data,scalex= True,color = 'r')
    plt.plot(time,summary_data, scalex= True,color = 'b')
    plt.legend(["compare", "summary"])
    plt.show()

def draw_figure_data (data):    # drawing the graph from a list, some time the data is not a path, it already in the code
    time = list()   # the volume in X
    volume = data # the all volume in Y
    p = 0

    row_len = len(volume)   # coding for pick some of the data in the .csv file
    final_vol = []
    while skip_step * p < row_len:
        time.append(p)
        final_vol.append(volume[skip_step * p])
        p += 1
       #use the file name to distinguish different data
    plt.plot(time, final_vol)

def draw_FFT_figure(data, sample_frequency):    # this fucntion will draw the FFT figure in both phase and amptitude in each frequency
    A = []
    P = []
    for i in range(len(data)):
        abs_value = abs(data[i])
        A.append(abs_value)
        phrase = np.arctan(data[i].imag/ data[i].real)
        P.append(np.degrees(phrase))

    plt.figure()
    plt.subplot(211)
    plt.xlabel('f')
    plt.ylabel('amptitude')
    plt.plot(sample_frequency, A)
    plt.subplot(212)
    plt.plot(sample_frequency, P)
    plt.xlabel('f')
    plt.ylabel('phrase')
    plt.show()


def draw_fft(data):
    real = []
    imag = []
    for each in data:
        real.append(each.real)
        imag.append(each.imag)
    plt.figure()
    plt.title("nyquist")
    plt.plot(real, imag)
    plt.show()


def filt_zero_data(x, y):
    x1=[]
    y1=[]
    for i in range(len(x)):
        if x[i] > 0.09375:
            x1.append(x[i])
            y1.append(y[i])
        else:
            continue
    return x1, y1


