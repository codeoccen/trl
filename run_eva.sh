#evaluate gate-finetune
CUDA_VISIBLE_DEVICES=3  python  evaluate.py \
  --dataset "gsm8k" \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --load_4bit true \
  --gate_or_adapter gate \
  --gate_weights "/data/liruizhe/gate_proj/gate_cot_result_2e-4/gate.pth"

#evaluate lora-finetune
CUDA_VISIBLE_DEVICES=1  python  evaluate.py \
  --dataset "gsm8k" \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --load_4bit true \
  --gate_or_adapter gate \
  --lora_weights "/data/liruizhe/gate_proj/gate_superni_result_2e-4/gate.pth"




CUDA_VISIBLE_DEVICES=1  python  evaluate.py \
  --dataset "gsm8k" \
  --base_model "/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec" \
  --load_4bit true \
  --gate_or_adapter gate \
  --gate_weights "/data/liruizhe/gate_proj/gate_flan_v2_result_2e-4/gate.pth"