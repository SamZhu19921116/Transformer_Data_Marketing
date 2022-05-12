#!/bin/bash
# -*- coding: UTF-8 -*-

# 需提前设置 data4transformer.json, training_transformer.json
# 也可以在此脚本中修改上述两个文件中的参数设置。为保持简便，仅列出数据源文件夹 data_dir 在此脚本中的设置方式
# data_dir="/disk1/assun/modelinfo/featureBuild/20220420/20220420-zaxd24-predictdata/"
data_dir=""

# work_dir='/home/sdsong/zwhuang14/multiloss_transformer/'
# conda_dir='/home/sdsong/sdsong/anaconda3/bin/activate'
# env_name='tf-ctr'

# 下面的路径是为177服务器上做测试
work_dir='/home/DataCastle/zwhuang14/multiloss_transformer/'
data_dir="../data/yqg-intent-addpack-posandneg2random/"
conda_dir="/home/DataCastle/anaconda3/bin/activate"
env_name="ml"

cd $work_dir
source $conda_dir $env_name
echo "converting traning/testing data..."
python3 src/data4transformer_multiloss.py --data_dir=$data_dir > log/data_log
echo "building and traning transformer..."
nohup python3 src/run_trans_multiloss.py >> log/train_log 2>&1 &