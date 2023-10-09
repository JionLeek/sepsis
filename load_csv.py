import pandas as pd
import numpy as np
import sys

negative_sample_limit = 10
negative_sample_limit = 10
sample_leap_count = 24

sample_seq_min_limit = 24
seq_length_hold = 24 * 5 + 50
seq_length = 24 * 5

cate_features = ["gcs_min","respiration","coagulation","liver","cardiovascular","cns","renal","respiration_24hours","coagulation_24hours","liver_24hours","cardiovascular_24hours","cns_24hours","renal_24hours"]
num_features = ["pao2fio2ratio_novent","pao2fio2ratio_vent","rate_epinephrine","rate_norepinephrine","rate_dopamine","rate_dobutamine","meanbp_min","uo_24hr","bilirubin_max","creatinine_max","platelet_min"]

def build_feature_buckets(df):
    sepsis_features_bucket = {}

    bucket_nums = {"pao2fio2ratio_novent":20,"pao2fio2ratio_vent":20,"rate_epinephrine":10,"rate_norepinephrine":20,"rate_dopamine":10,"rate_dobutamine":5,"meanbp_min":10,"uo_24hr":20,"bilirubin_max":10,"creatinine_max":10,"platelet_min":20}

    for field in num_features:
        df[field + '_rank'] = df[field].rank(method='first', ascending=True,na_option='bottom')
        cnt = df[field].count()
        bucket_num = bucket_nums[field]
        bucket_size = int(cnt / bucket_num)
        sepsis_features_bucket[field] = []
        for num in range(1,bucket_num):
            row_index = df.index[df[field + '_rank'] == num * bucket_size][0]
            sepsis_features_bucket[field].append(df.at[row_index, field])
        row_index = df.index[df[field + '_rank'] == cnt][0]
    return sepsis_features_bucket

def build_sample(records,sepsis_features_bucket):
    
    stay_id_2_feas = {}

    base_columns = ['sofa_24hours','label','hr']
    sepsis_columns = base_columns + num_features + cate_features
    sepsis_records = []

    for _,row in records.iterrows():
        new_row = []
        
        stay_id = int(row['stay_id'])
        hr = int(row['hr'])

        if stay_id not in stay_id_2_feas:
            stay_id_2_feas[stay_id] = {}

        stay_id_2_feas[stay_id][hr] = {}

        for col in base_columns:
            stay_id_2_feas[stay_id][hr][col] = int(row[col])

        for k,bucket_list in sepsis_features_bucket.items():
            curr_value = row[k]
            if np.isnan(curr_value):
                stay_id_2_feas[stay_id][hr][k] = -1
            else:
                bk_idx = 0
                #curr_value_float = float(curr_value)
                for bk in bucket_list:
                    if curr_value < bk:
                        break
                    bk_idx += 1
                stay_id_2_feas[stay_id][hr][k] = bk_idx

        for fea in cate_features:
            curr_value = row[fea]
            if np.isnan(curr_value):
                stay_id_2_feas[stay_id][hr][fea] = -1
            else:
                stay_id_2_feas[stay_id][hr][fea] = int(curr_value)


    for stay_id,records in stay_id_2_feas:
        sorted_records = sorted(records.items(), key=lambda x: x[0])
 
        seq_cnt = 0

        sample_dict = {}
        for field in sepsis_columns:
            sample_dict[field] = [-1 for i in range(seq_length_hold)]

        label = 0
        label_hr = 0

        for hr,feas in sorted_records:
            if seq_cnt > sample_seq_min_limit:
                pass
            
            label = feas['label'] 
            label_hr = hr

            if label > 0:
                break
            for field in sepsis_columns:
                sample_dict[field].append(feas[field])
                del sample_dict[field][0:1]

            seq_cnt += 1
        
        
    



def main(argv):
    df = pd.read_csv('../sepsis2.csv')
    sepsis_features_bucket = build_feature_buckets(df)

    build_sample(df,sepsis_features_bucket)
    
    #pd_output = pd.DataFrame(sepsis_records, columns=sepsis_columns)
    #pd_output.to_csv('sepsis_norm.csv')

    
if __name__ == '__main__':
    main(sys.argv)


