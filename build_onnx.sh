#!/bin/bash
# -*- coding: UTF-8 -*-

# 训练结束后，挑选最优模型转化为onnx
chkpt_dir=""

# work_dir='/home/sdsong/zwhuang14/multiloss_transformer/'
# conda_dir='/home/sdsong/sdsong/anaconda3/bin/activate'
# env_name='tf-ctr'

# 下面的路径是适用于177服务器上做测试
work_dir='/home/DataCastle/zwhuang14/multiloss_transformer/'
conda_dir="/home/DataCastle/anaconda3/bin/activate"
env_name="ml"

cd $work_dir 
source $conda_dir $env_name
echo "choosing a transformer model..."
model_file=$(python3 src/analysis_log.py --chkpt_dir=$chkpt_dir)
echo "the chosen best model is $model_file"
echo "converting the checkpoint to pb..."
pb_file=$(python3 src/tensorTopb.py --model_path=$model_file)
echo "converting $pb_file to onnx..."
onnx_file="${pb_file%.*}.onnx"
python3 -m tf2onnx.convert --input=$pb_file --inputs=input_ids_ph:0,input_mask_ph:0,is_training:0 --outputs=loss/Softmax:0 --output=$onnx_file --verbose --opset=11 >log/onnx_log 2>&1
echo "--------------------------------------------------------------------------"
echo "the final onnx file is at: $onnx_file ."
echo "--------------------------------------------------------------------------"
