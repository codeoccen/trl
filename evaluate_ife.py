import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_4bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()

    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """
    save_file = f'experiment_ife/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment_ife/mistralai')

    dataset = load_data(args)
    tokenizer, model = load_model(args)
    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []
    pbar = tqdm(total=total)
   
    for idx, data in enumerate(dataset):
        new_dic={}
        instruction = data.get('prompt')
        new_dic['prompt']=instruction
        outputs = evaluate(instruction)
        new_dic['response']=outputs
        output_data.append(new_dic)
       
        print(f'\rtest:{idx + 1}/{total}')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return

def generate_prompt_wo_cot(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
    

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        """ # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

        ### Instruction:
        Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

        ### Response:
        18.0

        ### Instruction:
        A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

        ### Response:
        3.0

        ### Instruction:
        Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

        ### Response:
        70000.0

        ### Instruction:
        {instruction}

        ### Response:
"""  # noqa: E501
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501
    
    
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/input_data.jsonl'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = read_jsonl(file_path)#json.load(open(file_path, 'r'))
    
    return json_data






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="google_ife")
    parser.add_argument('--model',default="mistralai/Mixtral-8x7B-Instruct-v0.1", choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B'])
    parser.add_argument('--adapter',default="gate_new", choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'])
    parser.add_argument('--base_model',default="/data/liruizhe/trans_1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/5c79a376139be989ef1838f360bf4f1f256d7aec")
    parser.add_argument('--lora', default=True)
    parser.add_argument('--lora_weights',default="/data/liruizhe/trans_result")#"/data/liruizhe/trans_result"
    parser.add_argument('--load_4bit', default=True)
    parser.add_argument('--gate', default=True)
    parser.add_argument('--gate_weights',default="/data/liruizhe/gate_alpaca_result/gate.pth")
    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_4bit = args.load_4bit
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model,cache_dir="/data/liruizhe/trans_1")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=load_4bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,cache_dir="/data/liruizhe/trans_1",bnb_4bit_compute_dtype=torch.float16,
        ) # fix zwq
        # if args.lora:
        #     model = PeftModel.from_pretrained(
        #         model,
        #         lora_weights,
        #         #torch_dtype=torch.float16,
        #         device_map={"":0}
        #     )
        if args.gate:
            custom_weights = torch.load(args.gate_weights)
            for name, param in model.named_parameters():
                if "gate" in name:
                    print(f"Replacing weights for: {name}")
                    print(torch.sum(custom_weights[name])) 
                    print(torch.sum(param.data)) 
                    param.data = custom_weights[name].to(torch.float16)
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     device_map={"": device},
        #     torch_dtype=torch.float16,
        # )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_4bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    fire.Fire(main)
