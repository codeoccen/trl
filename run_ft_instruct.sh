# Gate finetune on math_10_k 
CUDA_VISIBLE_DEVICES=0 python finetune_alpaca_code_cot.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path './alpaca.parquet' \
  --output_dir '/data/liruizhe/gate_proj/ni_alpaca_2e-4_512' \
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
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --data_path 'math_10k.json' \
  --output_dir '/data/liruizhe/gate_proj/test' \
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
  



# gate finetune on code_alpaca_20k 
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune_alpaca_code_cot.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
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


# finetune on code_alpaca_20k 
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune_alpaca_code_cot.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
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




  
CUDA_VISIBLE_DEVICES=3 python finetune_superni.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --data_path  "/home/rzhe/LLM-Adapters/code_alpaca_data.jsonl" \
  --output_dir '/data/liruizhe/gate_proj/gate_code_alpaca_result_2e-4_1epoch' \
  --batch_size 16 \
  --micro_batch_size 16 \
  --num_epochs 1 \
  --learning_rate 2e-4 \
  --cutoff_len 1024 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true

  
CUDA_VISIBLE_DEVICES=1 python finetune_superni.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "/home/rzhe/LLM-Adapters/flan_v2_data.jsonl" \
  --output_dir '/data/liruizhe/gate_proj/ni_gate_flan_result_2e-4_512_best_prompt_wram0' \
  --batch_size 64 \
  --micro_batch_size 64 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 512 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true



  
CUDA_VISIBLE_DEVICES=1 python finetune_superni.py \
  --base_model  "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "./flanv2.parquet" \
  --output_dir '/data/liruizhe/gate_proj/ni_gate_flan_parquet_2e-4_256_best_prompt_epoch1' \
  --batch_size 128 \
  --micro_batch_size 128 \
  --num_epochs 1 \
  --learning_rate 2e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true


CUDA_VISIBLE_DEVICES=1 python finetune_superni.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --data_path "/home/rzhe/LLM-Adapters/code_alpaca_data.jsonl" \
  --output_dir '/data/liruizhe/gate_proj/lora_code_alpaca_result_2e-4_1epoch' \
  --batch_size 16 \
  --micro_batch_size 16 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 1024 \
  --val_set_size 120 \
  --gate_or_adapter adapter \
  --use_gradient_checkpointing True\
  --load_4bit true


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192  finetune_share_gpt.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --data_path  "/home/rzhe/open_eval/sharegpt_split.json" \
  --output_dir '/data/liruizhe/gate_proj/gate_share_gpt_result_2e-5_3epoch_processed_gpt' \
  --batch_size 16 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-5\
  --cutoff_len 2048 \
  --val_set_size 20 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true





CUDA_VISIBLE_DEVICES=1 python finetune_share_gpt.py \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --data_path  "/home/rzhe/LLM-Adapters/ShareGPT_V3.json" \
  --output_dir '/data/liruizhe/gate_proj/ni_gate_share_gpt_result_2e-4_0.1' \
  --batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 0.1 \
  --learning_rate 2e-4 \
  --cutoff_len 2048 \
  --val_set_size 120 \
  --gate_or_adapter gate \
  --use_gradient_checkpointing True\
  --load_4bit true