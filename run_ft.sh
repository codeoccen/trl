# Gate finetune on alpaca
CUDA_VISIBLE_DEVICES=0 python finetune_alpaca_code_cot.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path './alpaca.parquet' \
  --output_dir '/data/liruizhe/gate_proj/no_ins_gate_alpaca' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true

# Lora finetune on alpaca
CUDA_VISIBLE_DEVICES=0 python finetune_alpaca_code_cot.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path './alpaca.parquet' \
  --output_dir '/data/liruizhe/gate_proj/no_ins_lora_alpaca' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter adapter \
  --adapter_name lora \
  --use_gradient_checkpointing True\
  --load_4bit true
  


###############################################



# Gate finetune on math_10_k 
CUDA_VISIBLE_DEVICES=0 python finetune_alpaca_code_cot.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path 'math_10k.json' \
  --output_dir '/data/liruizhe/gate_proj/no_ins_gate_cot' \
  --batch_size 32 \
  --micro_batch_size 32 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true

# Lora finetune on math_10_k
CUDA_VISIBLE_DEVICES=0 python finetune_alpaca_code_cot.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path 'math_10k.json' \
  --output_dir '/data/liruizhe/gate_proj/no_ins_lora_cot' \
  --batch_size 32 \
  --micro_batch_size 32 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --gate_or_adapter adapter \
  --adapter_name lora \
  --use_gradient_checkpointing True\
  --load_4bit true
  

###############################################


# gate finetune on code_alpaca_20k 
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune_alpaca_code_cot.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "/home/rzhe/LLM-Adapters/code_alpaca_20k.json" \
  --output_dir '/data/liruizhe/gate_proj/gate_code_alpaca_20k_result_2e-6_3epoch_256_wram0' \
  --batch_size 64 \
  --micro_batch_size 32 \
  --num_epochs 3 \
  --learning_rate 2e-6 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true


# lora finetune on code_alpaca_20k 
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune_alpaca_code_cot.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "/home/rzhe/LLM-Adapters/code_alpaca_20k.json" \
  --output_dir '/data/liruizhe/gate_proj/lora_code_alpaca_20k' \
  --batch_size 64 \
  --micro_batch_size 32 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --gate_or_adapter adapter \
  --adapter_name lora \
  --use_gradient_checkpointing True\
  --load_4bit true




###############################################


  
CUDA_VISIBLE_DEVICES=1 python finetune_flan_superni.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "./flanv2.parquet" \
  --output_dir '/data/liruizhe/gate_proj/no_ins_gate_flan' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true



  
CUDA_VISIBLE_DEVICES=1 python finetune_flan_superni.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "./flanv2.parquet" \
  --output_dir '/data/liruizhe/gate_proj/no_ins_lora_flan' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter adapter \
  --adapter_name lora \
  --use_gradient_checkpointing True\
  --load_4bit true






###############################################


  
CUDA_VISIBLE_DEVICES=1 python finetune_flan_superni.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "./super_ni_data.jsonl" \
  --output_dir '/data/liruizhe/gate_proj/no_ins_gate_super_ni' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true



  
CUDA_VISIBLE_DEVICES=1 python finetune_flan_superni.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "./super_ni_data.jsonl" \
  --output_dir '/data/liruizhe/gate_proj/no_ins_lora_super_ni' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter adapter \
  --adapter_name lora \
  --use_gradient_checkpointing True\
  --load_4bit true





###############################################



CUDA_VISIBLE_DEVICES=1 python finetune_share_gpt.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "/home/rzhe/LLM-Adapters/ShareGPT_V3.json" \
  --output_dir '/data/liruizhe/gate_proj/no_ins_gate_share_gpt' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 0.1 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true

CUDA_VISIBLE_DEVICES=1 python finetune_share_gpt.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "/home/rzhe/LLM-Adapters/ShareGPT_V3.json" \
  --output_dir '/data/liruizhe/gate_proj/no_ins_lora_share_gpt' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 0.1 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter adapter \
  --adapter_name lora \
  --use_gradient_checkpointing True\
  --load_4bit true