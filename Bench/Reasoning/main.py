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
"""The Reasoning Task""

import argparse
import logging
import torch
import random
import time
from tqdm import tqdm
import os
from utils import *


def log_data(text, path):
    with open(path + '/loggings.txt', 'a', encoding='utf-8') as f:
        f.write(text)
        print(text)
        f.write('\n')


def log_data_self(text, path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
        print(text)
        f.write('\n')


def log_start(MODEL, DATA, N, K, FN):
    log_name = MODEL + "_" + DATA + "_" + str(N) + "_" + str(K) + "_" + str(FN)
    try:
        os.mkdir('./log/' + log_name)
    except:
        log_name += time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('./log/' + log_name)

    with open('./log/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__), 'r', encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = './log/' + log_name
    return path


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = []
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum.append(val * n)
        self.count += n
        self.avg = sum(self.sum) / self.count


def main(decoder, dataset):
    args.dataset = dataset
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "meddialog":
        args.dataset_path = "./dataset/MedDialog/english-test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"

    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"

    path = log_start(args.model, args.dataset, args.N, args.K, args.method)
    log_data('*****************************', path)
    print(args)
    log_data('*****************************', path)
    fix_seed(args.random_seed)

    # Initialize decoder class (load model and tokenizer) ...

    log_data("setup data loader ...", path)
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == 'verifier_cot':
        demo_F = create_demo_text(args, cot_flag=True)
        demo_B = create_verifier_demo_text(args, cot_flag=True)
        demo_F = demo_F.split('\n\n')[:-1]
        demo_B = demo_B.split('\n\n')[:-1]
    elif args.method == 'few_shot_cot':
        demo = create_demo_text(args, cot_flag=True)
        print(1)

    correct_list = []
    tk = tqdm(dataloader)
    accs = AverageMeter()
    accs_sc = AverageMeter()
    accs_avg = AverageMeter()
    loggings = []


    demo_token_len = len(decoder.tokenizer.encode(demo))
    print('demo len:',demo_token_len)
    for i, data in enumerate(tk):
        log_data('*************************', path)
        log_data("{}st data".format(i + 1), path)

        # Prepare question template ...
        x, y = data

        if args.method == "few_shot_cot":

            x = [demo + "Q: " + i + "\n" + "A: "
                for i in x]

            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
            z = decoder.decode(args, x, max_length, i, 0, 1, '\n',demo_token_len=demo_token_len)

            # Answer extraction for zero-shot-cot ...
            for n in range(len(x)):
                pred = z[n]
                log_data(x[n] + pred, path)

                # Clensing of predicted answer ...
                pred = answer_cleansing(args, pred)

                # Choose the most frequent answer from the list ...
                log_data("pred : {}".format(pred), path)
                log_data("GT : " + y[n], path)
                log_data('*************************', path)

                # Checking answer ...

                correct = (np.array([pred]) == np.array([y[n]])).sum().item()
                correct_list.append(correct)

            tk.set_postfix(accs=(sum(correct_list) * 1.0 / len(correct_list)) * 100)

        else:
            raise ValueError("method is not properly defined ...")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / len(correct_list)) * 100
    log_data("accuracy : {}".format(accuracy), path)
    if args.method == 'verifier_cot':
        log_data('accs:{}  self_consistency:{}  Top_N_acc:{}'.format(accs.avg, accs_sc.avg, accs_avg.avg), path)
        loggings = pd.DataFrame(loggings)
        loggings.to_csv(path + '/ls.csv')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Reason with self-verification")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "bigbench_date", "object_tracking", "coin_flip", "last_letters", "meddialog"],
        help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=11, choices=[11],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="llama-2-7b",
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="few_shot_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "verifier_cot", "verifier"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=2048,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=4.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--N", type=int, default=32
    )
    parser.add_argument(
        "--K", type=int, default=0.3
    )
    parser.add_argument(
        "--FN", type=int, default=0, help="few-shot number"
    )

    args = parser.parse_args()

    return args

 
if __name__ == "__main__":
    args = parse_arguments()
    decoder = Decoder(args)
    for dataset in ["gsm8k", "commonsensqa", "addsub", "multiarith","aqua", "svamp", "singleeq", "bigbench_date", "strategyqa",
                    "coin_flip", "last_letters"]:
        main(decoder, dataset)
