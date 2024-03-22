#!/bin/bash

# CUDA_VISIBLE_DEVICES=0

# base_model="/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec"
# data_path='math_10k.json'
# output_dir='./trained_models/Mixtral-8x7B-Instruct-v0.1'
# batch_size=16
# micro_batch_size=2
# num_epochs=3
# learning_rate=2e-4
# cutoff_len=256
# val_set_size=120
# use_gradient_checkpointing=false
# load_4bit=true

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python finetune.py \
#   --base_model $base_model \
#   --data_path $data_path \
#   --output_dir $output_dir \
#   --batch_size $batch_size \
#   --micro_batch_size $micro_batch_size \
#   --num_epochs $num_epochs \
#   --learning_rate $learning_rate \
#   --cutoff_len $cutoff_len \
#   --val_set_size $val_set_size \
#   --use_gradient_checkpointing $use_gradient_checkpointing \
#   --load_4bit $load_4bit




CUDA_VISIBLE_DEVICES=3 python finetune.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --data_path 'math_10k.json' \
  --output_dir './trained_models/Mixtral-8x7B-Instruct-v0.1' \
  --batch_size 16 \
  --micro_batch_size 2 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --use_gradient_checkpointing \
  --load_4bit true \
  --gating_ft true


