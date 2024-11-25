import argparse
import json
import os
import re
import pdb
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import multiprocessing
import time
from process import extract_boxed_content, answer_process
import random

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--actor_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
    )
    parser.add_argument(
        "--dataset_name",
        nargs='+',
        default=["openai/gsm8k"],
    )
    parser.add_argument(
        "--dataset_type",
        nargs='+',
        default=["gsm8k"],
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--results_file", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct-selfimprove.json")
    parser.add_argument("--mode", type=str, default="inference")
    parser.add_argument("--test_know_answer", type=int, default=1)
    parser.add_argument("--need_false_data", type=int, default=0)
    parser.add_argument("--reserved_new_data", type=int, default=1)
    args = parser.parse_args()
    return args


def extract_answer(text, dataset_type):
    if dataset_type == "gsm8k":
        match = re.search(r'#### (-?[\d.,]+)', text)
        if match:
            text1 = match.group(1)
            text1 = text1.replace(",", "")
            return text1
        else:
            return None
    if dataset_type == "math":
        text1 = extract_boxed_content(text)
        text2 = answer_process(text1)
        return text2

def extract_response(text, dataset_type):
    temp = text.find("The answer is")
    if temp != -1:
        text = text[temp:]
        text = text[len("The answer is"):]
    try:
        text1 = extract_boxed_content(text)
        if text1 == None:
            text1 = text
            if dataset_type == "gsm8k":
                match = re.search(r'(-?[\d.,]+)', text1)
                if match:
                    text1 = match.group(1)
            if text1[-1] == ".":
                text1 = text1[0:-1]
        text2 = answer_process(text1)
        if dataset_type == "gsm8k":
            text2 = text2.replace(",", "")
    except Exception as e:
        text2 = text
    return text2

def safe_eval(expr):
    if re.match(r'^[\d\s+\-*/().]*$', expr):
        return eval(expr)
    else:
        raise ValueError("not an equation or number")

def compare_answer(answer, response, dataset_type):
    an = extract_answer(answer, dataset_type)
    re = extract_response(response, dataset_type)
    flag = False
    temp = True
    if an is not None and re is not None:
        try:
            if safe_eval(an) == safe_eval(re):
                flag = True
        except Exception as e:
            pass
        if an == re:
            flag = True
        if flag == False:
            temp = False
            temp_response = response[response.rfind("The answer is"):]
            if an in temp_response:
                temp = True
    return flag, temp

def generate_raw_template(instruction, model_name, dataset_type):
    if dataset_type == "gsm8k":
        if model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "Meta-Llama-3-8B-Instruct":
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction} Please also give your final number in the format of 'The answer is [your answer].'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's break it down step by step:\n\n"
        elif model_name == "meta-llama/Meta-Llama-3-8B" or model_name == "Meta-Llama-3-8B" or model_name == "meta-llama/Meta-Llama-32-1B" or model_name == "meta-llama/Meta-Llama-32-3B":
            prompt = f"Human: {instruction} Please also give your final number in the format of 'The answer is [your answer].'\n\nAssistant:Let's break it down step by step:\n\n"
        elif model_name == "meta-llama/Llama-2-13b-hf" or model_name == "Llama-2-13b-hf":
            prompt = f"Human: {instruction} Please also give your final number in the format of 'The answer is [your answer].'\n\nAssistant: Let's break it down step by step:\n\n"
        elif model_name == "meta-llama/Llama-2-13b-chat-hf" or model_name == "Llama-2-13b-chat-hf":
            prompt = f"<s> [INST] {instruction} Please also give your final number in the format of 'The answer is [your answer].' [/INST] Let's break it down step by step:\n\n"
    elif dataset_type == "math":
        if model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "Meta-Llama-3-8B-Instruct":
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's break it down step by step:\n\n"
        elif model_name == "meta-llama/Meta-Llama-3-8B" or model_name == "Meta-Llama-3-8B" or model_name == "meta-llama/Meta-Llama-32-1B" or model_name == "meta-llama/Meta-Llama-32-3B":
            prompt = f"Human: {instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\nAssistant:Let's break it down step by step:\n\n"
        elif model_name == "meta-llama/Llama-2-13b-hf" or model_name == "Llama-2-13b-hf":
            prompt = f"Human: {instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\nAssistant: Let's break it down step by step:\n\n"
        elif model_name == "meta-llama/Llama-2-13b-chat-hf" or model_name == "Llama-2-13b-chat-hf":
            prompt = f"<s> [INST] {instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.' [/INST] Let's break it down step by step:\n\n"
    return prompt

def generate_new_template(instruction, raw, critic, model_name, dataset_type):
    if dataset_type == "gsm8k":
        if model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "Meta-Llama-3-8B-Instruct":
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction} Please also give your final number in the format of 'The answer is [your answer].'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{raw}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI have made a step-by-step critic for your solution. Please read the following critic and generate a new answer to correct your solution. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's break it down step by step:\n\n"
        elif model_name == "meta-llama/Meta-Llama-3-8B" or model_name == "Meta-Llama-3-8B" or model_name == "meta-llama/Meta-Llama-32-1B" or model_name == "meta-llama/Meta-Llama-32-3B":
            prompt = f"Human: {instruction} Please also give your final number in the format of 'The answer is [your answer].'\n\nAssistant:{raw}\n<|end_of_text|>\nHuman: I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic}\n\nAssistant:Let's break it down step by step:\n\n"
        elif model_name == "meta-llama/Llama-2-13b-hf" or model_name == "Llama-2-13b-hf":
            prompt = f"Human: {instruction} Please also give your final number in the format of 'The answer is [your answer].'\n\nAssistant: {raw}\n</s>\nHuman: I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic}\n\nAssistant: Let's break it down step by step:\n\n"
        elif model_name == "meta-llama/Llama-2-13b-chat-hf" or model_name == "Llama-2-13b-chat-hf":
            prompt = f"<s> [INST] {instruction} Please also give your final number in the format of 'The answer is [your answer].' [/INST] {raw}</s><s> [INST] I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic} [/INST] Let's break it down step by step:\n\n"
    elif dataset_type == "math":
        if model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "Meta-Llama-3-8B-Instruct":
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{raw}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI have made a step-by-step critic for your solution. Please read the following critic and generate a new answer to correct your solution. Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\n{critic}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's break it down step by step:\n\n"
        elif model_name == "meta-llama/Meta-Llama-3-8B" or model_name == "Meta-Llama-3-8B" or model_name == "meta-llama/Meta-Llama-32-1B" or model_name == "meta-llama/Meta-Llama-32-3B":
            prompt = f"Human: {instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\nAssistant:{raw}\n<|end_of_text|>\nHuman: I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\n{critic}\n\nAssistant:Let's break it down step by step:\n\n"
        elif model_name == "meta-llama/Llama-2-13b-hf" or model_name == "Llama-2-13b-hf":
            prompt = f"Human: {instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\nAssistant: {raw}\n</s>\nHuman: I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\n{critic}\n\nAssistant: Let's break it down step by step:\n\n"
        elif model_name == "meta-llama/Llama-2-13b-chat-hf" or model_name == "Llama-2-13b-chat-hf":
            prompt = f"<s> [INST] {instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.' [/INST] {raw}\n</s><s> [INST] I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\n{critic} [/INST] Let's break it down step by step:\n\n"
    return prompt

def generate_responses(model, sampling_params, prompt):
    outputs = model.generate(prompt, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def batch_inference(
    inference_dataset,
    device_id,
    sampling_params,
    batch_size,
    actor_name,
    model_name,
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = LLM(model=actor_name, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results_true = []
    results_false = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = [
            generate_raw_template(item["question"], model_name, item["dataset_type"])
            for item in batch_items
        ]
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        for idx, data in enumerate(batch_items):
            question = data["question"]
            answer = data["answer"]
            response = batch_responses[idx]
            flag1, flag2 = compare_answer(answer, response, data["dataset_type"])
            new_data = {}
            new_data["idx"] = data["idx"]
            new_data["dataset_type"] = data["dataset_type"]
            new_data["question"] = question
            new_data["answer"] = answer
            new_data["actor_model"] = actor_name
            new_data["origin_response"] = "Let's break it down step by step:\n\n" + response
            new_data["correctness-origin"] = flag1
            if flag1 == True:
                results_true.append(new_data)
            else:
                results_false.append(new_data)
    return results_true, results_false


def inference_pipeline(
    actor_name,
    model_name,
    ds, 
    temperature, 
    sample_num, 
    results_path,
):
    total_gpu = 8
    stop_tokens = ["<|end_of_text|>", "</s>", "Human", "Assistant"]
    sampling_params = SamplingParams(max_tokens=1024, temperature=temperature, stop=stop_tokens)
    random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])
    true_data = []
    false_data = []

    with multiprocessing.Pool(processes=total_gpu) as pool:
        results = [
            pool.apply_async(batch_inference, args=(inference_dataset[device_id], device_id, sampling_params, 128, actor_name, model_name))
            for device_id in range(total_gpu)
        ]
        for r in results:
            result_true, result_false = r.get()
            true_data += result_true
            false_data += result_false
    with open(results_path, "a+") as g:
        for new_data in true_data:
            g.write(json.dumps(new_data))
            g.write("\n")
    with open("false_temp.json", "w") as g:
        json.dump(false_data, g, indent=2)

def shortcut_barrier(answer, origin_response, new_response):
    numbers_answer = len(re.findall(r'\d+', answer))
    numbers_origin = len(re.findall(r'\d+', origin_response))
    numbers_new = len(re.findall(r'\d+', new_response))
    if numbers_new < min(numbers_answer, numbers_origin) - 2:
        return False
    else:
        return True

def batch_new(
    inference_dataset,
    sampling_params,
    batch_size,
    actor_name,
    model_name,
    device_id,
    reserved_new_data,
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = LLM(model=actor_name, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results = []
    false_results = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = [
            generate_new_template(item["question"], item["origin_response"], item["critic"], model_name, item["dataset_type"])
            for item in batch_items
        ]
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        for idx, data in enumerate(batch_items):
            new_response = batch_responses[idx]
            answer = data["answer"]
            true_flag = False
            flag1, flag2 = compare_answer(answer, new_response, data["dataset_type"])
            if flag1 == True:
                if len(results) <= (reserved_new_data-1) or (len(results) > (reserved_new_data-1) and data["critic"] != results[-reserved_new_data]["critic"]):
                    if shortcut_barrier(data["answer"], data["origin_response"], new_response):
                        new_data = data.copy()
                        new_data["new_response"] = "Let's break it down step by step:\n\n" + new_response
                        new_data["correctness-new"] = flag1
                        results.append(new_data)
                        true_flag = True
            if not true_flag:
                new_data = data.copy()
                new_data["new_response"] = "Let's break it down step by step:\n\n" + new_response
                new_data["correctness-new"] = flag1
                false_results.append(new_data)
    return results, false_results


def new_pipeline(
    actor_name,
    model_name,
    temperature,
    sample_num,
    results_file,
    reserved_new_data,
    need_false_data = 0,
):
    with open("false_temp.json", "r") as g:
        ds = json.load(g)
    total_gpu = 8
    stop_tokens = ["<|end_of_text|>", "</s>", "Human", "Assistant"]
    sampling_params = SamplingParams(max_tokens=1024, temperature=temperature, stop=stop_tokens)
    random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])

    new_data = []
    false_new_data = []
    with multiprocessing.Pool(processes=total_gpu) as pool:
        results = [
            pool.apply_async(batch_new, args=(inference_dataset[device_id], sampling_params, 128, actor_name, model_name, device_id, reserved_new_data))
            for device_id in range(total_gpu)
        ]
        for r in results:
            new, false_new = r.get()
            new_data = new_data + new
            false_new_data = false_new_data + false_new
    with open("false_temp.json", "w") as g:
        json.dump(new_data + false_new_data, g, indent=2)
    with open(results_file, "a+") as g:
        for da in new_data:
            g.write(json.dumps(da))
            g.write("\n")
    if need_false_data:
        with open(results_file, "a+") as g:
            for da in false_new_data:
                g.write(json.dumps(da))
                g.write("\n") 

def batch_test(
    inference_dataset,
    device_id,
    sampling_params,
    batch_size,
    actor_name,
    model_name,
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = LLM(model=actor_name, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results_true = []
    results_false = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = [
            generate_raw_template(item["question"], model_name, item["dataset_type"])
            for item in batch_items
        ]
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        for idx, data in enumerate(batch_items):
            question = data["question"]
            answer = data["answer"]
            response = batch_responses[idx]
            flag1, flag2 = compare_answer(answer, response, data["dataset_type"])
            new_data = {}
            new_data["idx"] = data["idx"]
            new_data["dataset_type"] = data["dataset_type"]
            new_data["question"] = question
            new_data["answer"] = answer
            new_data["actor_model"] = actor_name
            new_data["origin_response"] = "Let's break it down step by step:\n\n" + response
            new_data["correctness-origin"] = flag1
            if flag1 == True:
                results_true.append(new_data)
            else:
                results_false.append(new_data)
    return results_true, results_false


def test_pipeline(
    actor_name,
    model_name,
    ds, 
    temperature, 
    sample_num, 
    results_path,
    test_know_answer = 1,
    need_false_data = 0,
):
    total_gpu = 8
    stop_tokens = ["<|end_of_text|>", "</s>", "Human", "Assistant"]
    sampling_params = SamplingParams(max_tokens=1024, temperature=temperature, stop=stop_tokens)
    random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])
    true_data = []
    false_data = []

    with multiprocessing.Pool(processes=total_gpu) as pool:
        results = [
            pool.apply_async(batch_test, args=(inference_dataset[device_id], device_id, sampling_params, 128, actor_name, model_name))
            for device_id in range(total_gpu)
        ]
        for r in results:
            result_true, result_false = r.get()
            true_data += result_true
            false_data += result_false
    if test_know_answer:
        with open(results_path, "a+") as g:
            for new_data in true_data:
                g.write(json.dumps(new_data))
                g.write("\n")
        with open("false_temp.json", "w") as g:
            json.dump(false_data, g, indent=2)
    else:
        false_data += true_data
        with open("false_temp.json", "w") as g:
            json.dump(false_data, g, indent=2)
        if need_false_data:
            with open(results_path, "a+") as g:
                for new_data in false_data:
                    g.write(json.dumps(new_data))
                    g.write("\n")



if __name__ == "__main__":
    args = arg_parse()
    actor_name = args.actor_name
    model_name = args.model_name
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    temperature = args.temperature
    sample_num = args.sample_num
    results_file = args.results_file
    mode = args.mode
    test_know_answer = args.test_know_answer
    need_false_data = args.need_false_data
    reserved_new_data = args.reserved_new_data
    if mode == "inference":
        ds = []
        for ds_name, ds_type in zip(dataset_name, dataset_type):
            if ds_type == "gsm8k":
                dataset = load_dataset(ds_name, "main", split = "train", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "math":
                dataset = load_dataset(ds_name, "all", split = "train", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["question"] = data.pop("problem")
                    data["answer"] = data.pop("solution")
                    data["dataset_type"] = ds_type
                    ds.append(data)
        inference_pipeline(actor_name, model_name, ds, temperature, sample_num, results_file)
    elif mode == "new":
        new_pipeline(actor_name, model_name, temperature, sample_num, results_file, reserved_new_data, need_false_data)
    elif mode == "test":
        ds = []
        for ds_name, ds_type in zip(dataset_name, dataset_type):
            if ds_type == "gsm8k":
                dataset = load_dataset(ds_name, "main", split = "test", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "math":
                dataset = load_dataset(ds_name, "all", split = "test", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["question"] = data.pop("problem")
                    data["answer"] = data.pop("solution")
                    data["dataset_type"] = ds_type
                    ds.append(data)
        test_pipeline(actor_name, model_name, ds, temperature, sample_num, results_file, test_know_answer, need_false_data)
