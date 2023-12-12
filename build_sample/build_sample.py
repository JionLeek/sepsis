import pandas as pd
import numpy as np
import json
import sys
import copy
from pathlib import Path

from collections import OrderedDict
g_bucket_num_limit = 20
g_bucket_cnt_limit = 70
g_default_bucket_num = 98
g_use_pre_feature_length_limit = 24

g_sample_seq_max_limit = 2900 
g_seq_min_length = 24 * 2 # 最短日志数量
g_seq_init_length = 24 * 3 #seq特征前置输出,用户补充seq长度不足问题
g_seq_length = g_seq_min_length + g_seq_init_length
g_seq_idx = [i for i in range(0,28,3)] + [(i*12 - 1)  for i in range(3,g_seq_length//12 + 1)] 

g_label_hrs = [24,48]
g_label_hrs_length = g_label_hrs[-1]
g_base_columns = ['label','hr','stay_id','charttime']
g_build_sample_step_num = 12
#构建特征离散化编码
def build_feature_buckets(df,num_features):
    sepsis_features_bucket = {}
    for field in num_features:
        df[field + '_rank'] = df[field].rank(method='first', ascending=True,na_option='bottom')
        cnt = df[field].count()
        bucket_num = g_bucket_num_limit
        if cnt < g_bucket_cnt_limit:
            bucket_num = cnt 
        bucket_size = int(cnt / bucket_num) 
        sepsis_features_bucket[field] = []
        for num in range(1,bucket_num):
            row_index = df.index[df[field + '_rank'] == num * bucket_size][0]
            if len(sepsis_features_bucket[field]) == 0 or sepsis_features_bucket[field][-1] + 1e-4 < df.at[row_index, field]:
                sepsis_features_bucket[field].append(df.at[row_index, field])
        row_index = df.index[df[field + '_rank'] == cnt][0]
    return sepsis_features_bucket

#生成样本通用部分
def gen_base_sample(sample_dict,i,subject_infos,seq_features):

    record_arr = [sample_dict['stay_id'][i],subject_infos["gender"],subject_infos['age']]
    for field in seq_features:
        record_arr.extend([sample_dict[field][i - j] for j in g_seq_idx])
    return ",".join(record_arr) + "\n"
        
#计算和生产样本
def extract_sample(sample_dict,subject_infos,seq_features):
    sample_len = len(sample_dict['hr'])
    sample_str_arr = []
    last_label_stay_id = '-1'

    i = g_seq_length-1
    while i < sample_len:
        curr_stay_id = sample_dict['stay_id'][i]
        # 一次住院已经就诊过一次脓毒症不重复抽取样本
        if sample_dict['label'][i] == '1':
            last_label_stay_id = curr_stay_id
        if last_label_stay_id == curr_stay_id:
            i += g_build_sample_step_num
            continue

        curr_seq_len = str(min(g_seq_length,i-g_seq_init_length+1))
        base_sample_str = gen_base_sample(sample_dict,i,subject_infos,seq_features)
        curr_label = "-1"
        label_idx = 1000 
        for j in range(1,g_label_hrs_length+1):
            if i + j + 1>= sample_len:
                break
            label_stay_id = sample_dict['stay_id'][i + j]
            #确保label和最后一次指标发生在同一次住院
            if label_stay_id != curr_stay_id:
                break
            label_idx = j
            curr_label = sample_dict['label'][i + j]
            #发现label=1后不额外抽取样本
            if curr_label == '1':
                break
        if curr_label == "-1":
            i += g_build_sample_step_num
            continue
        if curr_label == "1":
            for hr in g_label_hrs:
                if hr >= label_idx:
                    sample_str_arr.append(",".join([curr_label,str(hr),sample_dict['charttime'][i + j],curr_seq_len,base_sample_str]))
            i += g_build_sample_step_num
        else:
            #负样本
            for hr in g_label_hrs:
                sample_str_arr.append(",".join([curr_label,str(hr),sample_dict['charttime'][i + j],curr_seq_len,base_sample_str]))
            i += label_idx 

    return "".join(sample_str_arr)

def left_outer_join_log(records,features_bucket,subject_id_2_logs):
    str_default_bucket_num = str(g_default_bucket_num)  
    for _,row in records.iterrows():
        subject_id = str(int(row['subject_id']))
        charttime = row['charttime'].split(":")[0]

        if subject_id not in subject_id_2_logs or charttime not in subject_id_2_logs[subject_id]:
            continue

        for k,bucket_list in features_bucket.items():
            curr_value = row[k]
            if np.isnan(curr_value):
                subject_id_2_logs[subject_id][charttime][k] = str_default_bucket_num
            else:
                bk_idx = 0
                for bk in bucket_list:
                    if curr_value < bk:
                        break
                    bk_idx += 1
                subject_id_2_logs[subject_id][charttime][k] = str(bk_idx)

def left_outer_join_log_by_subject_id(records,features_bucket,subject_id_2_logs):
    str_default_bucket_num = str(g_default_bucket_num) 
    for _,row in records.iterrows():
        subject_id = str(int(row['subject_id']))
        if subject_id not in subject_id_2_logs:
            continue

        for k,bucket_list in features_bucket.items():
            curr_value = row[k]
            if np.isnan(curr_value):
                subject_id_2_logs[subject_id][k] = str_default_bucket_num
            else:
                bk_idx = 0
                for bk in bucket_list:
                    if curr_value < bk:
                        break
                    bk_idx += 1
                subject_id_2_logs[subject_id][k] = str(bk_idx)
    

def build_sepsis_log(records,stay_id_map,sepsis_features_bucket):

    subject_id_2_logs = {}
    base_columns = g_base_columns
    str_default_bucket_num = str(g_default_bucket_num)
    print("build sepsis base log.")
    not_subject_id_cnt = 0
    for _,row in records.iterrows():
        
        stay_id = str(int(row['stay_id']))
        if stay_id not in stay_id_map:
            subject_id = stay_id
            not_subject_id_cnt += 1
        else:
            subject_id = stay_id_map[stay_id]
        charttime = row['charttime'].split(":")[0]

        if subject_id not in subject_id_2_logs:
            subject_id_2_logs[subject_id] = OrderedDict()

        subject_id_2_logs[subject_id][charttime] = {}

        for col in base_columns:
            subject_id_2_logs[subject_id][charttime][col] = str(row[col])

        for k,bucket_list in sepsis_features_bucket.items():
            curr_value = row[k]
            if np.isnan(curr_value):
                subject_id_2_logs[subject_id][charttime][k] = str_default_bucket_num
            else:
                bk_idx = 0
                for bk in bucket_list:
                    if curr_value < bk:
                        break
                    bk_idx += 1
                subject_id_2_logs[subject_id][charttime][k] = str(bk_idx)
    print("not subject id cnt:{}".format(not_subject_id_cnt))
    return subject_id_2_logs

        
def revise_sample(subject_id_2_logs,subject_id_2_infos,seq_features):
    filename = Path("./datas/sepsis_sample_log.csv")
    sepsis_file = open(filename,'w')
    seq_feature_fields = []
    for fea in seq_features:
        seq_feature_fields.append(",".join([fea+"_"+str(i) for i in range(len(g_seq_idx))]))
    sepsis_file.write("label,hr,charttime,seq_length,stay_id,gender,age," + ",".join(seq_feature_fields)+ "\n")
    str_default_bucket_num = str(g_default_bucket_num)

    features_last_value = {}
    for field in seq_features:
        features_last_value[field + "_len"] = 0
    for id,records in subject_id_2_logs.items():
        if len(records) < g_seq_min_length:
            continue
        seq_cnt = 0
        sample_dict = {}
        for field in seq_features:
            sample_dict[field] = [str_default_bucket_num for i in range(g_seq_init_length)]
        for f in g_base_columns:
            sample_dict[f] = ['-1' for i in range(g_seq_init_length)]

        for _,feas in records.items():

            for f in g_base_columns:
                sample_dict[f].append(feas[f])

            for field in seq_features:
                pre_fea_value_len_name = field + "_len" 
                if field not in feas:
                    if features_last_value[pre_fea_value_len_name] == 0:
                        features_last_value[pre_fea_value_len_name] = 1
                        features_last_value[field] = str_default_bucket_num
                    else:
                        features_last_value[pre_fea_value_len_name] += 1
                        if features_last_value[pre_fea_value_len_name] > g_use_pre_feature_length_limit:
                            features_last_value[pre_fea_value_len_name] = 0
                    sample_dict[field].append(features_last_value[field])
                else:
                    if feas[field] == str_default_bucket_num:
                        if features_last_value[pre_fea_value_len_name] == 0:
                            features_last_value[pre_fea_value_len_name] = 1
                            features_last_value[field] = str_default_bucket_num
                        else:
                            features_last_value[pre_fea_value_len_name] += 1
                            if features_last_value[pre_fea_value_len_name] > g_use_pre_feature_length_limit:
                                features_last_value[pre_fea_value_len_name] = 0
                        sample_dict[field].append(features_last_value[field])
                    else:
                        features_last_value[pre_fea_value_len_name] = 1
                        features_last_value[field] = feas[field]
                        sample_dict[field].append(features_last_value[field])
            seq_cnt += 1
            if seq_cnt > g_sample_seq_max_limit:
                break
        #print("subject id:{}, len:{}".format(id,seq_cnt))
        stay_samples = extract_sample(sample_dict,subject_id_2_infos[id],seq_features)
        sepsis_file.write(stay_samples)
    
    sepsis_file.close()

def main(argv):

    # 获取subject_id,stay_id映射关系
    filename = Path("./datas/icustays.csv")
    icu_f = open(filename)
    line = icu_f.readline()
    line = icu_f.readline()
    stay_id_map = {}
    while line:
        l = line.strip().split(",")
        subject_id,stay_id = l[0],l[1]
        stay_id_map[stay_id] = subject_id
        line = icu_f.readline()
    icu_f.close()

    #print(stay_id_map)
    print("stay_id_map size:{}".format(len(stay_id_map)))
        
    filename = Path('./datas/age.csv') 
    age_df = pd.read_csv(filename)
    age_features = ["age"]
    age_features_bucket = build_feature_buckets(age_df,age_features)
    print("age_features_bucket:{}".format(age_features_bucket))
    
    filename = Path('./datas/sepsis.csv')
    sepsis_df = pd.read_csv(filename)
    # 需要对特征做离散化编码
    sepsis_features = ["pao2fio2ratio_novent","pao2fio2ratio_vent","rate_epinephrine","rate_norepinephrine","rate_dopamine","rate_dobutamine","meanbp_min","gcs_min","uo_24hr","bilirubin_max","creatinine_max","platelet_min"]
    seq_features = copy.deepcopy(sepsis_features)
    sepsis_features_bucket = build_feature_buckets(sepsis_df,sepsis_features)
    print("sepsis_features_bucket:{}".format(sepsis_features_bucket))
    subject_id_2_logs = build_sepsis_log(sepsis_df,stay_id_map,sepsis_features_bucket)

    subject_id_2_infos = {}
    default_bucket_num_str = str(g_default_bucket_num)
    for k in subject_id_2_logs.keys():
        subject_id_2_infos[k] = {"gender":default_bucket_num_str,"age":default_bucket_num_str}
    
    print("join log of gender.")
    filename = Path('./datas/gender.csv')
    gender_f = open(filename)
    line = gender_f.readline()
    line = gender_f.readline()
    while line:
        l = line.strip().split(",")
        subject_id,gender = l[0],l[1]
        line = gender_f.readline()
        if subject_id not in subject_id_2_infos:
            continue
        subject_id_2_infos[subject_id]['gender'] = gender
    gender_f.close()

    left_outer_join_log_by_subject_id(age_df,age_features_bucket,subject_id_2_infos)

    filename = Path('./table_conf.json')
    json_f = open(filename)
    json_conf = json.load(json_f)
    json_f.close()

    for k,conf in json_conf.items():
        filename = Path("./datas/{}.csv".format(k))
        df = pd.read_csv(filename)
        features_name = conf
        seq_features.extend(features_name)
        print("build {} feature bucket.".format(k))
        features_bucket = build_feature_buckets(df,features_name)
        print("{} features_bucket:{}".format(k,features_bucket))
        left_outer_join_log(df,features_bucket,subject_id_2_logs)

    print("raw log user cnt:{}".format(len(subject_id_2_logs)))
    print("seq_features:{}".format(seq_features))
    revise_sample(subject_id_2_logs,subject_id_2_infos,seq_features) 
    
if __name__ == '__main__':
    main(sys.argv)


