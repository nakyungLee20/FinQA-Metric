import json
import numpy as np
import re
from pre_data_fin import *

from models import *
import time
import torch.optim
from expressions_transfer import *
import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from train_and_evaluate import *


def load_fin_data(filename):  # load the json data to list(dict()) for FinQA dataset
    print("Reading FinQA lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data=[]
    for qa in data:
        qa_dict = {}
        qa_dict['original_text'] = qa['qa']['question']
        qa_dict['equation'] = "x=" + fin_to_math_eq(qa['qa']['program'])
        qa_dict['ans'] = qa['qa']['answer']
        out_data.append(qa_dict)

    return out_data

def cal_type(operator):
    oper = ""
    if operator == "divide":
        oper = "/"
    elif operator == "add":
        oper = "+"
    elif operator == "multiply":
        oper = "*"
    elif operator == "subtract":
        oper = "-"
    elif operator == "greater":
        oper = ">"
    elif operator == "exp":
        oper = "**"
    elif operator == "table_average":
        oper = " avg "
    elif operator == "table_max":
        oper = " max "
    elif operator == "table_min":
        oper = " min "
    elif operator == "table_sum":
        oper = " sum "

    return oper

def arg_to_num(text):
    # from FinQA/code/generator/utils
    # text = text.replace(",", "")
    try:
        num = float(text)
        num=str(num)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
                num = str(num)
            except ValueError:
                num = "n/a"
        elif "const_" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
            num = str(num)
        elif "none" in text:
            text = text.replace("none", "")
            num = text
        else:
            num = text

    return num


def fin_to_math_eq(fin_data):
    step = fin_data.split("), ")
    step_memory = []

    for idx in range(len(step)):
        if idx == len(step) - 1:
            step[idx] = step[idx][:len(step[idx]) - 1]

        func = step[idx].split("(")
        operator = cal_type(func[0])

        arg = func[1].split(", ")
        if re.search("#", arg[0]) and re.search("#", arg[1]):
            p1 = re.search("#", arg[0]).start()
            id1 = int((arg[0])[p1 + 1])
            arg[0] = step_memory[id1]
            p2 = re.search("#", arg[1]).start()
            id2 = int((arg[1])[p2 + 1])
            arg[1] = step_memory[id2]
        elif re.search("#", arg[0]):
            p = re.search("#", arg[0]).start()
            id = int((arg[0])[p + 1])
            arg[0] = step_memory[id]
        elif re.search("#", arg[1]):
            p = re.search("#", arg[1]).start()
            id = int((arg[1])[p + 1])
            arg[1] = step_memory[id]

        arg[0] = arg_to_num(arg[0])
        arg[1] = arg_to_num(arg[1])

        new_eq = "(" + arg[0] + operator + arg[1] + ")"
        step_memory.append(new_eq)

    return step_memory[-1]

# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y

def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split(" ")
        if "segmented_text" in d:
            seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 15:
            temp_g.append(g)
    print(temp_g)
    return pairs, temp_g, copy_nums

def main():
    fin_data = load_fin_data("../../../dataset/train.json")

    pairs, generate_nums, copy_nums = transfer_num(fin_data)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    # bert_tokenizer.add_tokens(['[num]'])
    bert_tokenizer.add_special_tokens({"additional_special_tokens":["[num]"]})
    count = 0
    new_items = []
    for item in pairs:
        old_token = item[0]
        sent = ""
        for token in old_token:
            if (token == 'NUM'):
                sent += " " + "[num]"
            else:
                sent += " " + token
        sent = "[CLS]" + sent + "[SEP]"
        new_token = bert_tokenizer.tokenize(sent)
        new_num_pos = [] # use bert to tokenize，numpos changed
        for i, token in enumerate(new_token):
            if (token == '[num]' or token == '[NUM]'):
                new_num_pos.append(i)
        if len(new_num_pos) != len(item[2]):
            print("new num error")
            print("old:", old_token)
            print("new:", new_token)
            count += 1
        new_item = {
            "tokens": new_token,
            "expression": item[1],
            "nums": item[2],
            "num_pos": new_num_pos
        }
        new_items.append(new_item)
    print(count)
    print(123)
    json.dump({"pairs": new_items, "generate_nums": generate_nums, "copy_nums": copy_nums}, open("../../../dataset/FinQA_mbert_token_train.json", "w"), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()

# fin_data = load_fin_data('_train.json')
# pairs, generate_nums, copy_nums = transfer_num(fin_data)

