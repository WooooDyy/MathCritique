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
import concurrent.futures
import random

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--critic_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--sample_num", type=int, default=1)
    args = parser.parse_args()
    return args

def critic_template(question, solution):
    if "Let's break it down step by step:\n\n" in solution:
        solution = solution[len("Let's break it down step by step:\n\n"):]
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nInstruction:
For a given question, there is a step-by-step solution, where each line is a step. Please check step by step to tell me whether the solution is correct or wrong. You should repeat each step and give me an explanation on whether this step is correct or wrong. Finally, you should tell me whether the final answer is correct or wrong.

Your answer format should be:

Step sentence: [Repeated sentence, ignoring new line break]
Correctness of the step: [Correct/Wrong]
Explanation: [Your Explanation]

Correctness of the final answer: [Correct/Wrong]
Explanation: [Your Explanation]

Question:
{question}

Solution:
{solution}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    
    return prompt

def critic_template_base(question, solution):
    if "Let's break it down step by step:\n\n" in solution:
        solution = solution[len("Let's break it down step by step:\n\n"):]
    instruction = f"""Instruction:
For a given question, there is a step-by-step solution, where each line is a step. Please check step by step to tell me whether the solution is correct or wrong. You should repeat each step and give me an explanation on whether this step is correct or wrong. Finally, you should tell me whether the final answer is correct or wrong.

Your answer format should be:

Step sentence: [Repeated sentence, ignoring new line break]
Correctness of the step: [Correct/Wrong]
Explanation: [Your Explanation]

Correctness of the final answer: [Correct/Wrong]
Explanation: [Your Explanation]


Question:
{question}

Solution:
{solution}

"""
    return instruction


def generate_responses(model, sampling_params, prompt):
    outputs = model.generate(prompt, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def batch_inference(
    inference_dataset,
    sampling_params,
    batch_size,
    critic_name,
    device_id,
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_id))
    model = LLM(model=critic_name, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = [
            critic_template(item["question"], item["origin_response"])
            for item in batch_items
        ]
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        for idx, data in enumerate(batch_items):
            new_data = data.copy()
            new_data["critic_model"] = critic_name
            new_data["critic"] = batch_responses[idx]
            results.append(new_data)
    return results


def critic_pipeline(
    critic_name, 
    temperature, 
    sample_num
):

    with open("false_temp.json", "r") as g:
        ds = json.load(g)
    total_gpu = 8
    stop_tokens = ["<|end_of_text|>", "</s>"]
    sampling_params = SamplingParams(max_tokens=1536, temperature=temperature, stop=stop_tokens)
    random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])
    
    critic_data = []
    # with multiprocessing.Pool(processes=total_gpu) as pool:
    #     results = [
    #         pool.apply_async(batch_inference, args=(inference_dataset[device_id], sampling_params, 128, critic_name, device_id))
    #         for device_id in range(total_gpu)
    #     ]
    #     for r in results:
    #         critc = r.get()
    #         critic_data = critic_data + critc
    with concurrent.futures.ProcessPoolExecutor(max_workers=total_gpu) as executor:
        futures = [
            executor.submit(batch_inference, inference_dataset[idx], sampling_params, 128, critic_name, device_id)
            for idx, device_id in enumerate([[0],[1],[2],[3],[4],[5],[6],[7]])
        ]       
        for future in concurrent.futures.as_completed(futures):
            critc = future.result()
            critic_data = critic_data + critc
    with open("false_temp.json", "w") as g:
        json.dump(critic_data, g, indent=2)


if __name__ == "__main__":
    args = arg_parse()
    critic_name = args.critic_name
    temperature = args.temperature
    sample_num = args.sample_num
    critic_pipeline(critic_name, temperature, sample_num)