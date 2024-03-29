# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""
from dataclasses import dataclass, field
import pdb
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments

import sys
print(sys.path)
sys.path.append('/home/rzhe/trl2')

from trl import ModelConfig, SFTTrainer, get_kbit_device_map, get_peft_config, get_quantization_config

import argparse
import os


tqdm.pandas()
from collections import OrderedDict

@dataclass
class ScriptArguments:
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_text_field: str = field(default="text", metadata={"help": "the text field of the dataset"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})

#'/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec'
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='trl')
    parser.add_argument("--model_name", type=str, default='/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec')#"mistralai/Mixtral-8x7B-Instruct-v0.1"
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")#"tatsu-lab/alpaca"trl-lib/ultrachat_200k_chatml
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--save_steps", type=int, default=200000)
    parser.add_argument("--output_dir", type=str, default="/data/liruizhe/gate_result/lr2e5")
    parser.add_argument("--use_peft", type=bool, default=False)
    parser.add_argument("--peft_lora_r", type=int, default=8)
    parser.add_argument("--peft_lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj","k_proj", "v_proj", "o_proj"])
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--gating_ft", type=bool, default=True)
    parser.add_argument("--cache_dir", type=str, default="/data/liruizhe/trans_1")
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=False)
    parser.add_argument("--mixed_precision", type=str, default="bf16")

    args = parser.parse_args()
    #args, model_config = parser.parse_args_into_dataclasses()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # logging_steps=args.logging_steps,
        # num_train_epochs=args.num_train_epochs,
        # max_steps=args.max_steps,
        # report_to=args.report_to,
        save_steps=args.save_steps,
        # save_total_limit=args.save_total_limit,
        # push_to_hub=args.push_to_hub,
        # hub_model_id=args.hub_model_id,
        # gradient_checkpointing=args.gradient_checkpointing,
        # TODO: uncomment that on the next release
        # gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
        #bf16=args.mixed_precision == "bf16"
    )
    #training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_config=ModelConfig(model_name_or_path=args.model_name,use_peft=args.use_peft,lora_r=args.peft_lora_r,lora_alpha=args.peft_lora_alpha,lora_target_modules=args.target_modules,load_in_4bit=args.load_in_4bit)
   

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        #revision=model_config.model_revision,
        trust_remote_code=True,
        #attn_implementation=model_config.attn_implementation,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        #low_cpu_mem_usage=args.low_cpu_mem_usage, 
        #device_map="auto",
        #use_cache=False if training_args.gradient_checkpointing else True,
        #device_map=get_kbit_device_map() if quantization_config is not None else None,
        #quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)#, use_fast=True,cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    #raw_datasets = load_dataset(args.dataset_name)load_dataset(args.dataset_name,cache_dir=args.cache_dir)#/home/rzhe/.cache/huggingface/datasets/tatsu-lab___alpaca
    raw_datasets =load_dataset('/home/rzhe/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017')#"/data/liruizhe/trans_data/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017")#"/home/rzhe/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017")
    train_dataset = raw_datasets["train"]
    
    #eval_dataset = raw_datasets["test"]

    ################
    # Training
    ################
    
    #pdb.set_trace()
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_config),
        gating_ft=args.gating_ft
    )
    trainer.train()
    # Step 6: Save the model
    if args.gating_ft:
        gate_state = OrderedDict()
        for k,v in trainer.model.state_dict().items():
            if "gate" in k:
                gate_state[k] = v
        torch.save(gate_state, training_args.output_dir+"/gate.pth")
    else:
        trainer.save_model(training_args.output_dir)
