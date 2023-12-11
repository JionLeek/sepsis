
import sys
import copy
from pathlib import Path

filename = Path("./datas/sepsis_sample_log.csv")
input_file = open(filename)
fields = input_file.readline()
line = input_file.readline()
filename = Path("./datas/train_sample_log.csv")
train_f = open(filename,'w')
train_f.write(fields)
filename = Path("./datas/eval_sample_log.csv")
eval_f = open(filename,'w')
eval_f.write(fields)

while line:
    l = line.strip().split(",")
    if l[2] < '2186-12-30':
        train_f.write(line)
    else:
        eval_f.write(line)
    line = input_file.readline()
input_file.close()
train_f.close()
eval_f.close()

