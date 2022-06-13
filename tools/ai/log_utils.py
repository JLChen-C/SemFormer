# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np
from tools.general.txt_utils import add_txt
import time
import math


def get_str_time(time):
    time_h = time // 3600
    time_m = (time - (time_h * 3600)) // 60
    time_s = time - ((time_h * 3600) + (time_m * 60))
    time_str = '{}h{}min{}sec'

    time_strs = []
    for time_value, time_scale in zip([time_h, time_m, time_s], ['h', 'min', 'sec']):
        if time_value > 0:
            time_strs += ['{:d}{}'.format(int(math.ceil(time_value)), time_scale)]
    if len(time_strs) == 0:
        time_str = '0sec'
    else:
        time_str = ':'.join(time_strs)
    return time_str

def get_time(t=None):
    if not t:
        t = time.localtime(time.time())
    return time.strftime("%Y-%m-%d, %H:%M:%S", t)

def nice_format(data, num_items_per_line=6):
    num_items = len(data)
    num_lines = num_items // num_items_per_line
    if num_lines * num_items_per_line < num_items:
        num_lines += 1
    keys = list(data.keys())
    format_string = '[i] ' + get_time() + ': '
    for l in range(1, num_lines + 1):
        if l > 1:
            format_string += '   '
        keys_this_line = keys[(l - 1) * num_items_per_line:l * num_items_per_line]
        for key in keys_this_line:
            val = data[key]
            if isinstance(val, float):
                format_string += ' {}:{:.4f}'.format(key, data[key])
            elif isinstance(val, int):
                format_string += ' {}:{:d}'.format(key, data[key])
            else:
                format_string += ' {}:{}'.format(key, data[key])
        format_string += '\n'
    return format_string


def log_print(message, path):
    """This function shows message and saves message.
    
    Args:
        pred_tags: 
            The type of variable is list.
            The type of each element is string.
        
        gt_tags:
            The type of variable is list.
            the type of each element is string.
    """
    print(message)
    add_txt(path, message)

class Logger:
    def __init__(self):
        pass

class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()
    
    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys
        
        dataset = [float(np.mean(self.data_dic[key])) for key in keys]
        if clear:
            self.clear()

        # if len(dataset) == 1:
        #     dataset = dataset[0]
            
        return dataset
    
    def clear(self):
        self.data_dic = {key : [] for key in self.keys}

