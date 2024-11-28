import argparse
import json
import os
import re
import pdb
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing
import time
from process import extract_boxed_content, answer_process
from collections import Counter

def arg_parse():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--results_file", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct-selfimprove.json")
    parser.add_argument("--mode", type=str, default="passk")
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

def compare_answer(answer, re, dataset_type):
    an = extract_answer(answer, dataset_type)
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
            temp_response = re
            if an in temp_response:
                temp = True
    return flag, temp

def sequential_data_filter():
    with open("false_temp.json", "r") as g:
        ds = json.load(g)
    for data in ds:
        data["origin_response"] = data["new_response"]
        data["correctness-origin"] = data["correctness-new"]
        del data["critic"]
        del data["critic_model"]
        del data["new_response"]
        del data["correctness-new"]
    with open("false_temp.json", "w") as g:
        json.dump(ds, g, indent=2)

def only_final_sequential(results_file):
    with open("false_temp.json", "r") as g:
        ds = json.load(g)
    with open(results_file, "w") as f:
        for data in ds:
            f.write(json.dumps(data))
            f.write("\n")

def response_extracter(ds, results_file):
    response_bucket = {}
    for data in ds:
        key = data["dataset_type"] + str(data["idx"])
        response_bucket[key] = []
    with open(results_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if "new_response" in data.keys():
                response = data["new_response"]
            else:
                response = data["origin_response"]
            response = extract_response(response, data["dataset_type"])
            key = data["dataset_type"] + str(data["idx"])
            response_bucket[key].append(response)
    return response_bucket
    
def pass_at_k(ds, response_bucket):
    accuracy = {}
    for data in ds:
        key = data["dataset_type"] + str(data["idx"])
        if data["dataset_type"] not in accuracy.keys():
            accuracy[data["dataset_type"]] = 0
        if key in response_bucket.keys():
            for item in response_bucket[key]:
                flag1, flag2 = compare_answer(data["answer"], item, data["dataset_type"])
                if flag1 == True:
                    accuracy[data["dataset_type"]] += 1
                    break
    return accuracy

def majority_vote(ds, response_bucket):
    accuracy = {}
    for data in ds:
        key = data["dataset_type"] + str(data["idx"])
        if data["dataset_type"] not in accuracy.keys():
            accuracy[data["dataset_type"]] = 0
        mv_response = Counter(response_bucket[key]).most_common(1)[0][0]
        flag1, flag2 = compare_answer(data["answer"], mv_response, data["dataset_type"])
        if flag1 == True:
            accuracy[data["dataset_type"]] += 1
    return accuracy

def accuracy_filter(accuracy):
    if "math" in accuracy.keys():
        accuracy["math"] = accuracy["math"] / 5000
    if "gsm8k" in accuracy.keys():
        accuracy["gsm8k"] = accuracy["gsm8k"] / 1319
    return accuracy

def test_file_sort(results_file):
    ds = []
    with open(results_file, "r") as f:
        for line in f:
            data = json.loads(line)
            ds.append(data)
    ds_list = sorted(ds, key=lambda x: (x['dataset_type'], x['idx']))

    with open(results_file, "w") as f:
        for data in ds_list:
            f.write(json.dumps(data))
            f.write("\n")

if __name__ == "__main__":
    args = arg_parse()
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    results_file = args.results_file
    mode = args.mode
    if mode == "sequential":
        sequential_data_filter()
    elif mode == "only_final_sequential":
        only_final_sequential(results_file)
    else:
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
        response_bucket = response_extracter(ds, results_file)
        if mode == "passk":
            accuracy = pass_at_k(ds, response_bucket)
        elif mode == "majority":
            accuracy = majority_vote(ds, response_bucket)
        accuracy = accuracy_filter(accuracy)
        print(accuracy)
        test_file_sort(results_file)
