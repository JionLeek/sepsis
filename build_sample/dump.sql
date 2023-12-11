\copy (select t1.stay_id as stay_id,t1.endtime as charttime, hr,pao2fio2ratio_novent,pao2fio2ratio_vent,rate_epinephrine,rate_norepinephrine,rate_dopamine,rate_dobutamine,meanbp_min,gcs_min,uo_24hr,bilirubin_max,creatinine_max,platelet_min,case when t2.stay_id is not null then 1 else 0 end  as label from sofa t1 left outer join sepsis3 t2 on t1.stay_id=t2.stay_id and t1.endtime=t2.sofa_time ) to '~/Downloads/sepsis.csv' with csv header ;

\copy (select distinct subject_id,stay_id from mimiciv_icu.icustays) to '~/Downloads/icustays.csv' with csv header ;

\copy (select subject_id,max(anchor_age) as age from age group by subject_id) to '~/Downloads/age.csv' with csv header ;

\copy (select subject_id,charttime,lactate,ph,baseexcess from bg ) to '~/Downloads/bg.csv' with csv header ;

\copy (select subject_id,charttime,inr,pt,ptt from coagulation ) to '~/Downloads/coagulation.csv' with csv header ;

\copy (select subject_id,charttime,crp from inflammation ) to '~/Downloads/inflammation.csv' with csv header ;

\copy (select subject_id,max(gender) as gender from icustay_detail group by subject_id ) to '~/Downloads/gender.csv' with csv header ;

\copy (select subject_id,charttime,wbc,rdw,platelet,hematocrit,hemoglobin,mchc from complete_blood_count ) to '~/Downloads/complete_blood_count.csv' with csv header ;

\copy (select subject_id,charttime,wbc as wbc2,basophils,eosinophils,lymphocytes,monocytes,neutrophils from blood_differential ) to '~/Downloads/blood_differential.csv' with csv header ;

\copy (select subject_id,charttime,aniongap,bicarbonate,bun,calcium,chloride,sodium,potassium from chemistry ) to '~/Downloads/chemistry.csv' with csv header ;

#要把每个辅助表单独下载到datas目录下，再通过脚本join关联