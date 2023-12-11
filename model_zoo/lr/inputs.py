# -*- coding:utf-8 -*-
import os
import numpy as np
import pickle
import time
import copy
from pathlib import Path
import pandas as pd
from torch import optim
from tqdm import tqdm
from collections import OrderedDict

def get_train_sample(args):
    print("load train sample. currtime:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),flush=True)
    filename = Path(args.train_file_path)
    input_data = pd.read_csv(filename,engine='c')

    return input_data


def get_eval_sample(args):
    print("load eval sample start. currtime:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),flush=True)
    filename = Path(args.eval_file_path)
    input_data = pd.read_csv(filename,engine='c')

    return input_data
