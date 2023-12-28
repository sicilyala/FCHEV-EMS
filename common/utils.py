import os
import sys

import scipy.io as scio
import xlrd

def get_driving_cycle(cycle_name):
    cycle_dir = "E:/SEU/Drive_Cycle/"
    filename = cycle_dir + cycle_name + '.mat'      # 须是行向量
    data = scio.loadmat(filename)
    speed_list = data['speed_vector'][0]            # class <ndarray>
    return speed_list

def get_driving_cycle_v2(cycle_name):
    cycle_dir = "E:/SEU/Data_Standard Driving Cycles/"
    filename = cycle_dir + cycle_name
    data_sheet = xlrd.open_workbook(filename + '.xls')
    table = data_sheet.sheets()[0]  # 0代表第一个工作表
    speed_list = table.col_values(0)  # class <list>        # speed list of leading car, be observed by rear car
    return speed_list

def get_acc_limit(speed_list, output_max_min=False):
    num = len(speed_list)
    acc_list = []
    for i in range(1, num):
        acc_list.append(speed_list[i]-speed_list[i-1])
    acc_list.append(0)
    if output_max_min:
        max_acc = max(acc_list)
        min_acc = min(acc_list)
        return acc_list, max_acc, min_acc
    else:
        return acc_list

class Logger:
    """
    save log automaticly
    """
    
    def __init__(self, filepath, filename, stream=sys.stdout):
        self.terminal = stream
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # create a note.txt
        notename = filepath+'note.txt'
        if not os.path.exists(notename):
            file = open(filepath+'note.txt', 'w')
            file.write('-----Configuration note-----'+'\n')
            file.close()
        
        self.log = open(filepath+filename, 'a')  # 文件末尾追加写入
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

