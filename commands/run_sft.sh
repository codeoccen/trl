#!/bin/bash
CUDA_VISIBLE_DEVICES=3
model_name="/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec"
dataset_name="tatsu-lab/alpaca"
batch_size=2
gradient_accumulation_steps=4
max_seq_length=512
learning_rate=2e-4
save_steps=200000
output_dir="/data/liruizhe/gate_result/lr2e5"
use_peft=false
peft_lora_r=8
peft_lora_alpha=32
load_in_4bit=true
gating_ft=true
cache_dir="/data/liruizhe/trans_1"
low_cpu_mem_usage=false
mixed_precision="bf16"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./examples/scripts/sft.py \
    --model_name "$model_name" \
    --dataset_name "$dataset_name" \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_seq_length $max_seq_length \
    --learning_rate $learning_rate \
    --save_steps $save_steps \
    --output_dir "$output_dir" \
    --use_peft $use_peft \
    --peft_lora_r $peft_lora_r \
    --peft_lora_alpha $peft_lora_alpha \
    --load_in_4bit $load_in_4bit \
    --gating_ft $gating_ft \
    --cache_dir "$cache_dir" \
    --low_cpu_mem_usage $low_cpu_mem_usage \
    --mixed_precision "$mixed_precision"
