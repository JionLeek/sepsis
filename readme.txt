一、 如果不做任何修改，只是跑模型
    1、datas 目录有有文件可以直接执行model_zoo下的train_runner.py训练模型
    2、如果datas目录下没有数据，需要先参考build_sample/dump.sql 内容，把数据dump到datas目录下，然后依次执行：
        build_sample.py -> train_test_split.py -> model_zoo/deepsepsis/train_runner.py 
    
二、 如果需要加特征
    1 ）先dump需要使用的视图或者表对应要用到的自断，下载数据到datas目录下， 格式保证有subject_id、charttime列，脚本会通过这两列做关联日志。
    2 ）接着配置table_conf.json 修改或者新增文件和对应特征字段
    3 ）然后依次执行：build_sample.py -> train_test_split.py -> model_zoo/deepsepsis/train_runner.py：


当前模型对比：
深度定制模型deepsepsis：
    AUC:0.7436713766124187
传统逻辑回归：
    AUC:0.6815936577134017
传统深度模型：
    AUC:0.7318255327853135
