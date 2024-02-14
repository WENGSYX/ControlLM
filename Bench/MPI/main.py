# https://github.com/WENGSYX/ControlLM
# Authors: Yixuan Weng (wengsyx@gmail.com)
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
# ==============================================================================
"""The MPI task"""


MODEL_PATH = "/path/to/alpaca/"
ITEMPATH = "../../inventories/mpi_120.csv"
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


template = """Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
"""

global_result = {}
global_cnt = {}

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm,trange
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pprint import pprint
from ControlLM import get_model

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

    for batch in range(0, len(questions), batch_size):
        outputs = model.generate(
            [prompt_to_tokens(tokenizer, system_prompt, template.format(prompt), 'Option') for prompt in
             questions[batch:batch + batch_size]],
            temperature=0.0,
            max_new_tokens=15,
            top_p=0.95,
            # top_k=0,
        )

        output_text = tokenizer.batch_decode(outputs)
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


def main():
    model, tokenizer = get_model(())
    dataset = getItems(ITEMPATH, TEST_TYPE)
    print("-" * 40)
    print(f"Current Prompt: {template}")
    results = []

    for dataset_type in ['E','N','O','C','A']:
        vec = torch.load("../Activate/vec_layer_78.pt".format(dataset_type))
        for num in trange(0,31):
            num = num/2-7.5
            model.reset_all()
            model.set_add_activations(78, num * vec.to(model.device))

            result, count = generateAnswer(tokenizer, model, dataset, template)

            mean_var = calc_mean_and_var(result)


            pprint(result)
            pprint(count)
            pprint(mean_var)
            result_file = {'Personality':dataset_type,'num':num,'result':result,'count':count,'mean_ver':mean_var}
            results.append(result_file)
    print('*******Finally:******')
    pprint(results)

if __name__ == "__main__":
    main()
