import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import Cutting
import Nyquist_preprocessing
import FFTrans


class cutting():
    def __init__(self):
        self.path = "/Users/zhengkaiwen/Desktop/CADEX/scanning/22-10-04/14_16_17/10_4_scanning2.csv"
        self.V = Cutting.pre_processing(self.path, 2)
        self.Va = Cutting.pre_processing(self.path, 1)
        self.A = Nyquist_preprocessing.get_current_logger(self.Va, 0.2)
        self.f = FFTrans.frequency

    def gradient_decent(self, start_pos, data):
        window_size = 1
        end_pos = start_pos + window_size
        t = 0.001 * window_size
        gradient = (data[start_pos] - data[end_pos])/t
        return gradient

    def change_detector(self, switch):
        res = []
        if switch == 'V':
            curve = self.V
        if switch == 'A':
            curve = self.A
        else:
            print('switch error, data was not detected')
        n = 0
        p = 0
        g = []
        start_pos = 1
        while start_pos < len(curve):
            gradient = (curve[start_pos] - curve[start_pos-1]) / 0.001
            if gradient == 0:
                p += 1
            elif len(g) == 0 and gradient != 0:
                g.append(gradient)
            else:
                if np.sign(gradient) == np.sign(g[-1]):
                    g.append(gradient)
                else:
                    g.pop()
            n += 1
            if len(g) != 0 and p > 1:
                value = stats.mode(curve[start_pos-n: start_pos])[0][0]
                rec = [start_pos-n, start_pos, value]
                res.append(rec)
                g = []
                p = 0
                n = 0
            start_pos += 1
        return res


    def step_cancelling(self, switch):
        result = []
        res = []
        value_record = self.change_detector(switch)
        for each in value_record:
            new_record = [each[2] for _ in range(each[1] - each[0])]
            res.extend(new_record)

        init_value = res[0]
        value_amount = [init_value]
        amount = 0
        index = 0
        for i in range(1, len(res)):
            if len(value_amount) > 2:
                record = [init_value, index, amount]
                init_value = res[i-1]
                index = i
                value_amount = [init_value]
                amount = 0
                result.append(record)
            if res[i] not in value_amount:
                value_amount.append(res[i])
                amount += 1
            else:
                amount += 1
        return result

    def pick_up_points(self, switch):
        result = []
        res = []
        value_record = self.step_cancelling(switch)
        for each in value_record:
            new_record = [each[0] for _ in range(each[1], each[1] + each[2])]
            res.extend(new_record)
        init_val = res[0]
        for i in range(1, len(res)):
            if res[i] != init_val:
                record = [i-1, init_val]
                init_val = res[i]
                result.append(record)
            else:
                pass
        return result

    def reset_value(self, switch):
        res = self.pick_up_points(switch)
        x= []
        v = []
        for each in res:
            x.append(each[0])
            v.append(each[1])
        plt.figure()
        plt.title('step canceling')
        plt.plot(self.V)
        plt.plot(x, v)
        # plt.plot(self.V)
        plt.show()
        return x, v

    def gradient(self, switch):
        x, y = self.reset_value(switch)
        init_x = x[0]
        init_y = y[0]
        gradient = []
        for i in range(1,len(x)):
            gradient.append((y[i]-init_y)/(x[i]- init_x))
            init_y = y[i]
            init_x = x[i]
        x.pop(-1)
        plt.figure()
        plt.title('Gradient result for the signal')
        plt.plot(x, gradient)
        plt.show()
        return gradient

    def find_init_start_position(self, switch):
        gradient = self.gradient(switch)
        x, v = self.reset_value(switch)
        result = []
        memo_up = []
        memo_down = []
        p = 0
        possible_pos = None
        for i in range(len(gradient)-1):
            if gradient[i] < 0 and gradient[i + 1] > 0:
                memo_up.append(i)
            if gradient[i] > 0 and gradient[i + 1] < 0:
                memo_down.append(i)
