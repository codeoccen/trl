CUDA_VISIBLE_DEVICES=2 nohup python -u evaluate.py \
  --dataset "gsm8k" \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --load_4bit true \
  --gate true >gate_result_gsm8k_16