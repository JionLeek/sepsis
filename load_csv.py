import pandas as pd
import numpy as np
import sys
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


def main(argv):
    df = pd.read_csv('../sepsis2.csv')
    sepsis_features_bucket = build_feature_buckets(df)

    base_columns = ['stay_id','hr','sofa_24hours','label']
    sepsis_columns = base_columns + num_features + cate_features
    sepsis_records = []

    for _,row in df.iterrows():
        new_row = []
        for col in base_columns:
            new_row.append(int(row[col]))

        for k,bucket_list in sepsis_features_bucket.items():
            curr_value = row[k]
            if np.isnan(curr_value):
                new_row.append(-1)
            else:
                bk_idx = 0
                #curr_value_float = float(curr_value)
                for bk in bucket_list:
                    if curr_value < bk:
                        break
                    bk_idx += 1
                new_row.append(bk_idx)

        for fea in cate_features:
            new_row.append(row[fea])

        sepsis_records.append(new_row.copy())
        
    
    pd_output = pd.DataFrame(sepsis_records, columns=sepsis_columns)
    pd_output.to_csv('sepsis_norm.csv')
                
    
if __name__ == '__main__':
    main(sys.argv)


