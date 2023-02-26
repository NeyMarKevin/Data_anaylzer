import sys

import numpy as np
from PyQt5.QtWidgets import *   # a package in the folder to treat the data and plot it
import matplotlib.pyplot as plt

import FFTrans
import Nyquist_preprocessing
import logger_cutting
import noise_canceling_method
import read_scanning_file
import reconstruction
import signal_test
import stratgy


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.paths = "" # initial file path
        self.setWindowTitle("drag the file to here")
        self.resize(500, 400)    #define the window size
        self.textBrowser = QTextBrowser()
        self.setCentralWidget(self.textBrowser) # the main project of the window is textBrowser
        self.setAcceptDrops(True)
        self.btn = QPushButton("clear", self)
        self.btn.setGeometry(200,300,100,30)
        self.btn.clicked.connect(self.when_btn_click)

    def when_btn_click(self):       #clear the path and the graph
        self.textBrowser.setText("")    # clear the showing path
        plt.close("all")            #close the figure

    def dragEnterEvent(self, event):
        num_channel = 3
        file = event.mimeData().urls()[0].toLocalFile() # get the location of the dragging file (in str)
        print("draged files ===> {}".format(file))  # plot it to the window
        self.paths = file + "\n"                    # save it in a list
        self.textBrowser.setText(self.paths)        # show in the GUI window
        event.accept()
        # reconstruction.get_impedance().get_R('data_logger')
        # reconstruction.get_impedance().get_R('eis')
        # logger_cutting.cutting().find_init_start_position('V')
        # reconstruction.nyquist().getR_in_all_f()
        reconstruction.logger_data().calculate_long()

        # current_A = stratgy.get_final_A_eis(file, [1, 2])
        # current_V = stratgy.get_final_V_eis(file, [2])
        # R, para= stratgy.get_final_R(file, [2], [1])
        # mini_wave_cancel_noise
        # noise_canceling_method.main_miniwave(file, 2)
        # signal_test.eis_Data().curve_fit()

app = QApplication(sys.argv) # main code for open and close the window.

window = Window()
window.show()
sys.exit(app.exec_())