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
"""The Language_Modeling task"""


import json
import os
from ControlLM.llama import get_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from tqdm import trange
import torch

batchsize = 6

def get_data(model, tokenizer, dataset):
    datas = []
    if dataset == 'lambada':
        data_file = "lambada_test_plain_text.txt"
        data = open(data_file).read().splitlines()
        for idx, text in enumerate(data):
            datas.append(text)
    elif dataset == 'pile':
        data_file = 'pile_10k.json'
        datas = json.load(open(data_file))

    elif dataset == 'bbh':
        data_file = 'cot-prompts'
        for file in os.listdir(data_file):
            datas.append(open(data_file + '/' + file).read()[111:])
    elif dataset == 'WikiMIA':
        data_file = 'wikimia.json'
        datas = json.load(open(data_file))

    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    seq_len = []
    for i in trange(len(datas)):
        prompt, answer = datas[i], datas[i]
        a_ids = tokenizer.encode(text=prompt[:20000], add_special_tokens=True)

        seq_length = len(a_ids)

        input_id = a_ids.copy()
        label = a_ids.copy()

        input_ids = input_id + (1024 - seq_length) * [0]
        labels = label + (1024 - seq_length) * [-100]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["labels"].append(labels)

        seq_len.append(seq_length)
    return model_inputs, seq_len


def get_acc_ppl(model, tokenizer, model_inputs, seq_lenght):
    ppl = []
    accs = []
    for index in trange(0, min(len(model_inputs['input_ids']), 1024), batchsize):
        input_ids = model_inputs["input_ids"][index:index + batchsize]
        labels = model_inputs["labels"][index:index + batchsize]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor(input_ids).to(model.device)).logits

            labels = torch.tensor(labels, device=model.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, 32000)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            ppl.append(torch.exp(loss).item())

            for idx in range(len(input_ids)):
                ids = logits[idx, :seq_lenght[index + idx]].argmax(1)
                lab = torch.tensor(labels[idx])[:seq_lenght[index + idx]]
                acc = sum(ids[:-1] == lab[1:]).item() / len(ids[:-1])
            accs.append(acc)

    return ppl, accs


def print_with_save(text, texts):
    print(text)
    texts.append(text)
    with open('result_ppl_7b_10.txt'
            , 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    return texts


def get_vec(layer):
    return torch.load(f"../../Activate/vec_layer_{layer}.pt")


def main():
    layer = 10
    model, tokenizer = get_model()
    data = {}
    seq_len = {}
    for dataset in ['lambada', 'pile', 'bbh', 'WikiMIA']:
        data[dataset], seq_len[dataset] = get_data(model, tokenizer, dataset)

    texts = []
    for i in range(0, 21, 1):
        num = i / 10 - 1

        model.reset_all()
        vec = get_vec(layer)
        model.set_add_activations(layer, num * vec.to(model.device))

        for dataset in ['lambada', 'pile', 'bbh', 'WikiMIA']:
            ppl, accs = get_acc_ppl(model, tokenizer, data[dataset], seq_len[dataset])
            print_with_save(f'llama_7b_chat Change  {dataset} {str(num)} PPL:' + str(sum(ppl) / len(ppl)), texts)
            print_with_save(f'llama_7b_chat Change {dataset} {str(num)} ACC:' + str(sum(accs) / len(accs)), texts)
            print_with_save(f'', texts)
        print_with_save(f'', texts)

if __name__ == "__main__":
    main()
