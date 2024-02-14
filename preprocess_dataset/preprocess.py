import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pprint import pprint
from activate_model import get_model as loadModel
from datasets import Dataset

MODEL_PATH = "/path/to/alpaca/"
ITEMPATH_1k = "../inventories/mpi_1k.csv"
ITEMPATH = "../inventories/mpi_120.csv"
TEST_TYPE = None
SCORES = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
}


system_prompt = "You are a helpful, honest and concise assistant."

def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
    dialog_tokens = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"

    return dialog_tokens


template1 = """Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
"""


template2 = """Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(1). False
(2). True
"""

template3 = """Given a statement of you: 'You {}.'
Are you confirm if you agree with this statement?"""

global_result = {}
global_cnt = {}


device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
global_result = {}
global_cnt = {}


def getItems(filename=ITEMPATH, item_type=None):
    data = pd.read_csv(filename)
    if item_type is not None:
        items = data[data["instrument"] == item_type]
    else:
        items = data
    return items




def generateAnswer(tokenizer, model, dataset, template, scores=SCORES):
    global_result = {}
    global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
    batch_size = 24
    questions, answers = [], []
    for _, item in dataset.iterrows():
        questions.append(item["text"].lower())

    for batch in range(0,len(questions),batch_size):

        outputs = model.generate(
            [prompt_to_tokens(tokenizer, system_prompt, template.format(prompt), 'Option') for prompt in questions[batch:batch+batch_size]],
            temperature=0.0,
            max_new_tokens=20,
            top_p=0.95,
            # top_k=0,
        )

        output_text = tokenizer.batch_decode(outputs)
        print(output_text[0])
        print('***********')
        answer = [text.split("[/INST]")[-1] for text in output_text]
        answers.extend(answer)

    answer_number = 0
    for _, item in dataset.iterrows():
        label = item["label_ocean"]
        key = item["key"]
        parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answers[answer_number][:12], flags=0)
        if parsed_result:
            parsed_result = parsed_result.group()[0].upper()

            score = scores[parsed_result]
            if label not in global_result:
                global_result[label] = []

            global_cnt[parsed_result] += 1
            if key == 1:
                global_result[label].append(score)
            else:
                global_result[label].append(6 - score)
        else:
            global_cnt["UNK"] += 1
        answer_number += 1
    return global_result, global_cnt


def calc_mean_and_var(result):
    mean = {}
    std = {}
    for key, item in result.items():
        mean[key] = np.mean(np.array(item))
        std[key] = np.std(np.array(item))

    return {
        "mean": list(sorted(mean.items(), key=lambda item: item[0])),
        "std": list(sorted(std.items(), key=lambda item: item[0])),
    }

def get_activate():
    from datasets import load_dataset
    datasets = load_dataset('WENGSYX/ControlLM_Personalities')['Openness'].to_list()
    model.get_and_save_activations(data=datasets, save_path='./Activate/Openness')

def main():
    dataset = getItems(ITEMPATH, TEST_TYPE)
    print("-" * 40)
    print(f"Current Prompt: {template3}")

    datas = {}
    question_120 = []
    for _, item in dataset.iterrows():
        question_120.append(item["text"].lower())

    for dataset_type in ['E', 'N', 'O', 'C', 'A']:
        ds = []
        dataset = getItems(ITEMPATH_1k, TEST_TYPE)
        number = 0
        for _, item in dataset.iterrows():
            if item['label_ocean'] == dataset_type and item['text'].lower() not in question_120:
                question = template3.format(item["text"].lower())
                if item['key'] == 1:
                    answer_matching_behavior = 'Yes'
                    answer_not_matching_behavior = 'No'
                    number += 1
                else:
                    answer_matching_behavior = 'No'
                    answer_not_matching_behavior = 'Yes'

                ds.append({'question': question, 'answer_matching_behavior': answer_matching_behavior,
                           'answer_not_matching_behavior': answer_not_matching_behavior})

        datas[{'E':'Extraversion','N':'Neuroticism','C':'Conscientiousness','A':'Agreeableness','O':'Openness'}[dataset_type]] = []
        n = 0
        if number < len(ds) / 2:
            for d in ds:
                if d['answer_matching_behavior'] == 'No':
                    if n < number:
                        datas[{'E':'Extraversion','N':'Neuroticism','C':'Conscientiousness','A':'Agreeableness','O':'Openness'}[dataset_type]].append(d)
                        n += 1
                else:
                    datas[{'E':'Extraversion','N':'Neuroticism','C':'Conscientiousness','A':'Agreeableness','O':'Openness'}[dataset_type]].append(d)
        else:
            for d in ds:
                if d['answer_matching_behavior'] == 'Yes':
                    if n < (len(ds) - number):
                        datas[{'E':'Extraversion','N':'Neuroticism','C':'Conscientiousness','A':'Agreeableness','O':'Openness'}[dataset_type]].append(d)
                        n += 1
                else:
                    datas[{'E':'Extraversion','N':'Neuroticism','C':'Conscientiousness','A':'Agreeableness','O':'Openness'}[dataset_type]].append(d)


        personalities = ['Extraversion','Neuroticism','Conscientiousness','Agreeableness','Openness']
        for personal in personalities:
            ds = Dataset.from_list(datas[personal])
            ds.push_to_hub('xxxx/xxxx',personal)



if __name__ == "__main__":
    main()
