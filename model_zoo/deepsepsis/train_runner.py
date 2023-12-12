

import os
import json
import sys
import copy
from pathlib import Path

#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from torch import nn
import numpy as np
import pickle
from config import set_args
import pandas as pd
from torch import optim
from model import DeepSepsis
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from inputs import get_eval_sample,get_train_sample

def evaluate_model(model,valid_loader):
    # 指定多gpu运行
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(valid_loader)):
            features, label = x[0].to(device), x[1].to(device)
            logits = model(features)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        return cur_auc,valid_preds


def train_model(model,train_loader):

    # 指定多gpu运行
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        model.cuda()

    loss_fct = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    loss_fct.to(device)
    print("train start. currtime:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),flush=True)

    total_auc = 0.0
    cur_auc = 0.5
    calc_auc_num = 0
    for epoch in range(args.Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        valid_labels, valid_preds = [], []
        for step, x in enumerate(train_loader):
            features, label = x[0].to(device), x[1].to(device)
            pred = model(features)
            pred = pred.view(-1)
            loss = loss_fct(pred, label)
            valid_labels.extend(label.cpu().numpy().tolist())
            valid_preds.extend (pred.view(-1).data.cpu().numpy().tolist())
            if len(valid_preds) > 5000:
                cur_auc = roc_auc_score(valid_labels, valid_preds)
                total_auc += cur_auc
                calc_auc_num += 1
                valid_labels.clear()
                valid_preds.clear()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if ((step + 1) % 10 == 0 or (step + 1) == len(train_loader)) and calc_auc_num > 0:
                print("Epoch {:04d} | Step {:04d} / {} | Total Auc  {:.4f} | Current Auc {:.4f} | Total Loss {:.4f} | Current Batch Loss {:.4f} | Time {:.4f}".format(
                    epoch+1, step+1, len(train_loader),total_auc/calc_auc_num,cur_auc, train_loss_sum/(step+1), loss.cpu().item(), time.time() - start_time))
        scheduler.step()

if __name__ == '__main__':
    args = set_args()

    filename = Path('./model_zoo/feature_conf.json')
    json_f = open(filename)
    feature_json = json.load(json_f)
    json_f.close()

    feature_conf = feature_json['base_feature']
    for fea,voc_size in feature_json['seq_feature'].items():
        for i in range(feature_json['seq_len']):
            feature_conf[fea+"_"+str(i)] = voc_size

    train_data = get_train_sample(args)

    print("DataLoader. currtime:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),flush=True)
    train_dataset = TensorDataset(torch.LongTensor(train_data[feature_conf.keys()].values),
                                  torch.FloatTensor(train_data['label'].values),)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)

    model = DeepSepsis(feature_conf,num_hidden_units=[512,256],n_layers=2)

    print("training. currtime:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),flush=True)
    train_model(model,train_loader)

    filename = Path(args.model_path)
    torch.save(model.state_dict(),filename,_use_new_zipfile_serialization=False)

    eval_data= get_eval_sample(args)
    eval_dataset = TensorDataset(torch.LongTensor(eval_data[feature_conf.keys()].values),
                                  torch.FloatTensor(eval_data['label'].values))
    valid_loader = DataLoader(dataset=eval_dataset, batch_size=args.eval_batch_size, shuffle=False)


    model.load_state_dict(torch.load(args.model_path))
    print("evaling. currtime:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),flush=True)
    auc,valid_preds = evaluate_model(model,valid_loader) 
    print("T+1 Model eval AUC:{}".format(auc))
    eval_data['preds'] = np.array(valid_preds)
    filename = Path(args.predict_file_path)
    eval_data.to_csv(filename,columns=feature_json["output_columns"])