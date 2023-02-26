import json
import base64
import gzip
import tkinter.messagebox

from matplotlib import pyplot as plt
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import *
from statistics import stdev


## MAKE CHANGES HERE ####
# Add experiments to be plotted in experiment_list

nyquist_title = 'Nyquist Plot'
voltage_title = ''


class Database:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)

    def get_model_names(self):
        return self.conn.execute('SELECT NAME from BATTERYMODELS').fetchall()

    def get_batteries_by_model(self, model_name):
        id = self.get_model_id(model_name)
        return self.conn.execute('SELECT SERIAL_NUMBER FROM BATTERIES WHERE MODEL_ID=?', (id[0][0],)).fetchall()

    def get_master_experiments_by_battery(self, battery_serial_number):
        id = self.get_battery_id(battery_serial_number)
        return self.conn.execute('SELECT ID, EXPERIMENT_NAME, START_DATETIME FROM MASTEREXPERIMENT WHERE BATTERY_ID=?',
                                 (id[0][0],)).fetchall()

    def get_experiment_step_by_master_id(self, master_id):
        return self.conn.execute('SELECT * FROM EXPERIMENT_STEP WHERE MASTERID=?', (master_id,)).fetchall()

    def get_data_by_step_id(self, step_id):
        return self.conn.execute('SELECT * FROM DATA WHERE STEPID=?', (step_id,)).fetchall()

    def get_soh_by_step_id(self, step_id):
        return self.conn.execute('SELECT SOH FROM SOH WHERE STEPID=?', (step_id,)).fetchall()

    def get_eis_by_step_id(self, step_id):
        return self.conn.execute('SELECT * FROM EIS WHERE STEPID=?', (step_id,)).fetchall()

    def get_model_id(self, serial_number):
        return self.conn.execute('SELECT * from BATTERYMODELS WHERE NAME=?', (serial_number,)).fetchall()

    def get_battery_id(self, serial_number):
        return self.conn.execute('SELECT ID from BATTERIES WHERE SERIAL_NUMBER=?', (serial_number,)).fetchall()


def get_info(id, dict_key, data):
    return_list = []
    experiment_step = data.get_experiment_step_by_master_id(id)
    try:
        step_id = experiment_step[0][0]
    except IndexError:
        tk.messagebox.showinfo(message='Error: The experiment #{} does not exist in the database, try again with a valid experiment number'.format(id))
        quit()
    eis = data.get_eis_by_step_id(step_id)
    try:
        info = json.loads(eis[0][2])
    except IndexError:
        tk.messagebox.showinfo(message='Error: The experiment #{} does not contian any data, try again with a valid experiment'.format(id))
        quit()
    for i in info[dict_key]:
        return_list.append(i*1000)
    return return_list


def get_volt_current(id, dict_key, data):
    info_list = []
    experiment_step = data.get_experiment_step_by_master_id(id)
    for index, exp in enumerate(experiment_step):
        if exp[2] == 'EIS':
            try:
                step_id = experiment_step[index][0]
                eis = data.get_eis_by_step_id(step_id)
            except IndexError:
                tk.messagebox.showinfo(message='Error: The experiment #{} does not exist in the database, try again with a valid experiment number'.format(id))
            try:
                info = json.loads(eis[0][3])
                compressed_waveform = info['data']
                waveform = json.loads(gzip.decompress(base64.b64decode(compressed_waveform)))
                info_list.append(waveform[dict_key])
            except IndexError:
                tk.messagebox.showinfo(message='Error: The experiment #{} does not contian any data, try again with a valid experiment'.format(id))
                info = []
                info_list.append(info)
    return info_list


def plot_nyquist(i):
    plt.plot(np.array(real), np.array(imaginary), label=i)
    plt.title(nyquist_title)
    plt.legend(loc='best')


def plot_voltage_current(i, volt, current):
    plt.subplot(2, 1, 1)
    for j in volt:
        plt.plot(np.array(j), label=i)
    plt.title('Voltage')
    plt.subplot(2, 1, 2)
    for j in current:
        plt.plot(np.array(j), label=i)
    plt.title('Current')
    plt.legend(loc='best')


# Ask user which database they want to opem
# db_file = fd.askopenfilename()
# data = Database(db_file)

# Plotting Nyquist - comment back in if you want to plot it

# Plot Nyquist
# for i in experiment_list:
#     real = get_info(i, 'Real')
#     imaginary = get_info(i, '-Imaginary')
#     plot_nyquist(i)
# plt.grid(True)
# plt.show()

# Plot Voltage or Current
def get_va_from_db(experiment_list):
    db_file = fd.askopenfilename()
    data = Database(db_file)
    for i in experiment_list:
        current = get_volt_current(i, 'Current', data)
        volt = get_volt_current(i, 'Voltage', data)
    return current, volt

