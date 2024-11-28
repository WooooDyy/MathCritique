# MathCritique
Implementation for the research paper "[Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision](http://arxiv.org/abs/2411.16579)".

<p align="center">
  📃 <a href="http://arxiv.org/abs/2411.16579" target="_blank">Paper</a > • 🌐 <a href="https://mathcritique.github.io" target="_blank">Project Page</a > • 🤗 <a href="https://huggingface.co/datasets/MathCritique/MathCritique-76k" target="_blank">MathCritique-76k</a > <br>
</p >



## Introduction
Training large language models (LLMs) to spend more time thinking and reflection before responding is crucial for effectively solving complex reasoning tasks in fields such as science, coding, and mathematics. However, the effectiveness of mechanisms like self-reflection and self-correction depends on the model’s capacity to accurately assess its own performance, which can be limited by factors such as initial accuracy, question difficulty, and the lack of external feedback. 

In this paper, we delve into **a two-player paradigm** that separates the roles of reasoning and critique models, where the critique model provides step-level feedback to supervise the reasoning (actor) model during both test-time and train-time. We first propose **AutoMathCritique**, an automated and scalable framework for collecting critique data, resulting in a dataset of responses paired with step-level feedback **(MathCritique-76k**). 

Fine-tuning language models with this dataset enables them to generate natural language feedback for mathematical reasoning. We demonstrate that the critique models consistently improve the actor’s performance on difficult queries at test- time, especially when scaling up inference-time computation. 

Motivated by these findings, we introduce the critique-based supervision to the actor’s self-training process, and propose a **critique-in-the-loop self-improvement method**. Experiments show that the method improves the actor’s exploration efficiency and solution diversity, especially on challenging queries, leading to a stronger reasoning model. 

Lastly, we take the preliminary step to explore training **self-talk reasoning models via critique data** and showcase its potential.



## Project Structure
Below is the structure of the repository:
```
MathCritique/
│
├── LLaMA-Factory/          # The latest version of the Llamafactory repository 
|                           # only modifies dataset_info.json.
│
├── selfimprove/            # The folder containing all the training and testing files
│   ├── critic.py           # Used to generate critics by critique models
│   ├── data_filter.py      # Used to integrate test files during training
│   ├── inference.py        # Used to generate responses by actor models (both origin and refined)
│   ├── process.py          # Answer extracter
│   ├── test_filter.py      # Used to process test files during testing
│   ├── inference-all.sh    # The main script for training an actor model by self-improve
│   └── evaluate-all.sh     # The main script for testing an existing model by different methods
│
│
├── README.md               # Documentation (this file)
└── LICENSE                 # License file
```
## Usage
### Install Dependencies
```bash
# install LLaMA-Factory dependencies
cd LLaMA-Factory
pip3 install -e ".[torch,metrics]"

# install vllm for inference
pip3 install vllm

# install deepspeed for training
pip3 install deepspeed==0.15.4

# the newest llama-factory has a bug with transformers, so we need to install a custom transformers version,
# you can see this issus https://github.com/huggingface/transformers/issues/34503#issuecomment-2448933790
pip3 install git+https://github.com/techkang/transformers.git

```
### Run experiment
The selfimprove/inference-all.sh file is the core script of this experiment, encompassing the entire process of training, inference, and evaluation. Below is an explanation of some key configuration parameters in this script:

``` bash
dataset_name="lighteval/MATH openai/gsm8k"  # Path of the dataset
dataset_type="gsm8k math"  # Name of the dataset
sample_num=5  # Number of sampling iterations
reserved_new_data=1  # Number of new samples to retain
temperature=0.7  # Sampling temperature
model_name="meta-llama/Meta-Llama-3-8B"  # Model name
model_type="Base"  # Model type
template="default"  # Llama3 default template (llamafactory type)
ITER_NUM=3  # Number of iterations
EXP_NUM="1origin"  # Experiment identifier
actor_model_name=${model_name}  # Path to the actor model
USE_CRITIC=0  # Whether to use a critic (0 = No, 1 = Yes)
USE_ORIGINAL=1  # Whether to use the original dataset (0 = No, 1 = Yes)
USE_PREVIOUS=0  # Whether to include data from previous rounds (0 = No, 1 = Yes)
USE_SELFIMPROVE_FEEDBACK=1  # Whether to use self-improvement feedback (0 = No, 1 = Yes)
USE_GPT4_FEEDBACK=1  # Whether to use GPT-4 feedback (0 = No, 1 = Yes)
critic_model_name="meta-llama/Meta-Llama-3-8B-Instruct" # Path to the critique model

```
Just Run 
```
bash selfimprove/inference-all.sh
```

## Dataset introduction
For iteration 0, we will finetune an actor model directly by using our dataset in the /selfimprove/meta-llama

The origin data is constructed by using our prompt on the training set of GSM8k and MATH. Each query will contain an question and its corresponding answer.

The new data is constructed by using our prompt on GPT4 feedback data. Each query will contain an question, a feedback and its corresponding refined answer. Note that we release 100 examples for each dataset here.

We will release more data later.


## License:
[Apache2.0 License](https://github.com/hotdog-zz/Mathcritique/blob/main/License.txt)

## Contact

Zhiheng Xi: [zhxi22@m.fudan.edu.cn](zhxi22@m.fudan.edu.cn)

## Citation
Please cite the paper if you use our data, model or code.
```
@misc{xi2024enhancingllmreasoningcritique,
      title={Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision}, 
      author={Zhiheng Xi and Dingwen Yang and Jixuan Huang and Jiafu Tang and Guanyu Li and Yiwen Ding and Wei He and Boyang Hong and Shihan Do and Wenyu Zhan and Xiao Wang and Rui Zheng and Tao Ji and Xiaowei Shi and Yitao Zhai and Rongxiang Weng and Jingang Wang and Xunliang Cai and Tao Gui and Zuxuan Wu and Qi Zhang and Xipeng Qiu and Xuanjing Huang and Yu-Gang Jiang},
      year={2024},
      eprint={2411.16579},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.16579}, 
}
```
