一、 如果不做任何修改，只是跑模型
    1、datas 目录有有文件可以直接执行model_zoo下的train_runner.py训练模型
    2、如果datas目录下没有数据，需要先参考build_sample/dump.sql 内容，把数据dump到datas目录下，然后依次执行：
        build_sample.py -> train_test_split.py -> model_zoo/deepsepsis/train_runner.py 
    
二、 如果需要加特征
    1 ）先dump需要使用的视图或者表对应要用到的字段，下载数据到datas目录下， 格式保证有subject_id、charttime列，脚本会通过这两列做关联日志。
    2 ）接着配置table_conf.json 修改或者新增文件和对应特征字段
    3 ）然后依次执行：build_sample.py -> train_test_split.py -> model_zoo/deepsepsis/train_runner.py：

三、 样本特征细节
    1） 指标部分
        a、对缺失特征使用24内的最近指标作为补充,如果要修改可以修改build_sample.py的参数g_use_pre_feature_length_limit
        b、对特征全部做离散化处理，正常映射到20以内的bucket桶内
        c、因为a的特点，样本很多指标是24小时内的，为了尽量避免重复特征，样本会分别选取近0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 35, 47, 59, 71, 83, 95, 107, 119小时的指标（每个指标都安a方式处理）作为序列特征
        d、为了尽量避免重复取样，如果一个病人的指标记录连续产生负样本，会每隔12小时取一次样
    2） 生成完样本后可以在train_test_split.py脚本里修改训练和测试样本的切分方式，比如2186-12-30前为训练样本，之后为测试样本，严格按照T+1方式评估
    3） 样本label暂时只判断24、48小时内脓毒症，如果有需要可以修改build_sample.py脚本 g_label_hrs参数

当前模型对比：
预估24小时内脓毒症：
    深度定制模型deepsepsis：
        AUC:0.7747445532555094
    传统逻辑回归：
        AUC:0.742782367035355
    传统深度模型：
        AUC:0.7543802064469395

