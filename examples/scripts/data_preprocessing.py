def alpaca_process(data, val_set_size=0):
    # def tokenize(prompt, add_eos_token=True):
    #     # there's probably a way to do this with the tokenizer settings
    #     # but again, gotta move fast
    #     result = tokenizer(
    #         prompt,
    #         truncation=True,
    #         max_length=cutoff_len,
    #         padding=False,
    #         return_tensors=None,
    #     )
    #     if (
    #             result["input_ids"][-1] != tokenizer.eos_token_id
    #             and len(result["input_ids"]) < cutoff_len
    #             and add_eos_token
    #     ):
    #         result["input_ids"].append(tokenizer.eos_token_id)
    #         if "chatglm" not in base_model:
    #             result["attention_mask"].append(1)

    #     result["labels"] = result["input_ids"].copy()

    #     if "chatglm" in base_model:
    #         return {"input_ids": result["input_ids"], "labels": result["labels"]}
    #     else:
    #         return result
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        # tokenized_full_prompt = tokenize(full_prompt)
        # if not train_on_inputs:
        #     user_prompt = generate_prompt({**data_point, "output": ""})
        #     tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        #     user_prompt_len = len(tokenized_user_prompt["input_ids"])

        #     tokenized_full_prompt["labels"] = [
        #                                             -100
        #                                         ] * user_prompt_len + tokenized_full_prompt["labels"][
        #                                                             user_prompt_len:
        #                                                             ]  # could be sped up, probably
        return {'text':full_prompt}
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )

    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    return train_data, val_data
    

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501