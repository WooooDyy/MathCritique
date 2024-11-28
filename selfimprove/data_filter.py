from openai import OpenAI
import json
import pdb
from datasets import load_dataset
import re
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
    )
    parser.add_argument("--model_type", type=str, default="Instruct")
    parser.add_argument("--inference_file", type=str)
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--sft_file", type=str)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--use_original_data", type=int, default=0)
    parser.add_argument("--use_previous_data", type=int, default=0)
    parser.add_argument("--use_critic_data", type=int, default=0)
    parser.add_argument("--use_selfimprove_feedback", type=int, default=1)
    parser.add_argument("--use_gpt4_feedback", type=int, default=1)
    args = parser.parse_args()
    return args

def input_template(question, dataset_type):
    if dataset_type == "gsm8k":
        prompt = f"""{question} Please also give your final number in the format of 'The answer is [your answer].'"""
    elif dataset_type == "math":
        prompt = f"""{question} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'"""
    return prompt

def input_new_template(instruction, raw, critic, new, dataset_type, model_name):
    if dataset_type == "gsm8k":
        if model_name == "meta-llama/Meta-Llama-3-8B":
            prompt = f"{instruction} Please also give your final number in the format of 'The answer is [your answer].'\n\nAssistant:{raw}\n<|end_of_text|>\nHuman: I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic}"
        elif model_name == "meta-llama/Llama-2-13b-hf":
            prompt = f"{instruction} Please also give your final number in the format of 'The answer is [your answer].'\n\nAssistant: {raw}\n</s>\nHuman: I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic}"
    elif dataset_type == "math":
        if model_name == "meta-llama/Meta-Llama-3-8B":
            prompt = f"{instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\nAssistant:{raw}\n<|end_of_text|>\nHuman: I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\n{critic}"
        elif model_name == "meta-llama/Llama-2-13b-hf":
            prompt = f"{instruction} Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\nAssistant: {raw}\n</s>\nHuman: I have made a step-by-step critic for your solution. Please read the following critic and use this critic as a reference to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Please also give your final number in the format of 'The answer is $\\boxed{{[your answer]}}$.'\n\n{critic}"
    return prompt

def critic_template(question, solution):
    if "Let's break it down step by step:\n\n" in solution:
        solution = solution[len("Let's break it down step by step:\n\n"):]
    prompt = f"""Instruction:
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
{solution}"""
    
    return prompt

def data_filter(model_name, model_type, inference_file, results_file, use_selfimprove_feedback):
    generate_data = []
    with open(results_file, "w") as g:
        with open(inference_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if data["correctness-origin"] == True:
                    solution = data["origin_response"]
                elif data["correctness-new"] == True:
                    solution = data["new_response"]
                
                dataset_type = data["dataset_type"]
                question = data["question"]
                if model_type == "Instruct":
                    human = input_template(question, dataset_type)
                    generate_one = {"conversations": []}
                    generate_one["conversations"].append({"from": "human", "value": human})
                    generate_one["conversations"].append({"from": "gpt", "value": solution})
                    generate_data.append(generate_one)
                elif model_type == "Base":
                    input = input_template(question, dataset_type) + "\n"
                    generate_one = {"instruction": input, "input": "", "output": solution}
                    generate_data.append(generate_one)

                if use_selfimprove_feedback:
                    if data["correctness-origin"] == False and data["correctness-new"] == True:
                        origin = data["origin_response"]
                        critic = data["critic"]
                        new = data["new_response"]
                        if model_type == "Instruct":
                            pass
                        elif model_type == "Base":
                            input = input_new_template(question, origin, critic, new, dataset_type, model_name) + "\n"
                            generate_one = {"instruction": input, "input": "", "output": new}
                            generate_data.append(generate_one)
        json.dump(generate_data, g, indent=2)

def sft_data(sft_file, use_original_data, use_previous_data, use_critic_data, iter, use_gpt4_feedback):
    sftdata = []
    file_name = sft_file[:-5]
    if use_original_data or iter == 0:
        with open(file_name + "_0_origin.json", "r") as f:
            data = json.load(f)
        sftdata = sftdata + data
        if use_gpt4_feedback:
            with open(file_name + "_0_new.json", "r") as f:
                data = json.load(f)
            sftdata = sftdata + data
    if iter > 0:
        if use_previous_data:
            for i in range(1, iter+1):
                with open(file_name + f"_{i}_origin.json", "r") as f:
                    data = json.load(f)
                sftdata = sftdata + data
                if use_critic_data:
                    with open(file_name + f"_{i}_new.json", "r") as f:
                        data = json.load(f)
                    sftdata = sftdata + data
        else:
            with open(file_name + f"_{iter}_origin.json", "r") as f:
                data = json.load(f)
            sftdata = sftdata + data
            if use_critic_data:
                with open(file_name + f"_{iter}_new.json", "r") as f:
                    data = json.load(f)
                sftdata = sftdata + data
    print(len(sftdata))
    with open(sft_file, "w") as f:
        json.dump(sftdata, f, indent=2)

def critic_data(model_name, inference_file, results_file):
    generate_data = []
    results_file = results_file[:-5]
    results_file += "critic.json"
    with open(results_file, "w") as g:
        with open(inference_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if "correctness-new" in data.keys():
                    if data["correctness-new"] == True:
                        solution = data["origin_response"]
                        critic = data["critic"]
                        question = data["question"]
                        human = critic_template(question, solution)
                        generate_one = {"conversations": []}
                        generate_one["conversations"].append({"from": "human", "value": human})
                        generate_one["conversations"].append({"from": "gpt", "value": critic})
                        generate_data.append(generate_one)
        json.dump(generate_data, g, indent=2)
    

if __name__ == "__main__":
    args = arg_parse()
    model_type = args.model_type
    inference_file = args.inference_file
    results_file = args.results_file
    iter = args.iter
    model_name = args.model_name
    use_selfimprove_feedback = args.use_selfimprove_feedback
    use_gpt4_feedback = args.use_gpt4_feedback
    if iter > 0:
        data_filter(model_name, model_type, inference_file, results_file, use_selfimprove_feedback)
        critic_data(model_name, inference_file, results_file)
    sft_file = args.sft_file
    if sft_file is not None:
        use_original_data = args.use_original_data
        use_previous_data = args.use_previous_data
        use_critic_data = args.use_critic_data
        sft_data(sft_file, use_original_data, use_previous_data, use_critic_data, iter, use_gpt4_feedback)
