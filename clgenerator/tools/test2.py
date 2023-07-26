import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.pre_data_fin import *
from src.expressions_transfer import *
from contra_pair import *

import json
import tqdm
import random

def main():
    data1 = json.load(open("../dataset/FinQA_mbert_token_train.json"))["pairs"]
    data2 = json.load(open("../dataset/MathQA_mbert_token_train.json"))["pairs"]

    pairs = []
    data_pairs = []
    cnt_0 = 0

    tree_ls1 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data1]
    tree_ls2 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data2]
    for _ in tree_ls1:
        add_order(_, 0)
    for _ in tree_ls2:
        add_order(_, 0)

    for i, t1 in enumerate(tqdm.tqdm(tree_ls1)):
        for j, t2 in enumerate(tree_ls2):
            if j <= i:
                continue
            if comp(t1, t2):
                pairs.append([i,j,0,0])

    for i, t1 in enumerate(tqdm.tqdm(tree_ls1)):
        for j, t2 in enumerate(tree_ls2):
            x = is_sub_tree(t1, t2)
            if x != -1:
                pairs.append([i,j,0,x])
            if x != 0:
                x = is_sub_tree(t2, t1)
                if x != -1:
                    pairs.append([j,i,x,0])

    json.dump(pairs, open("../dataset/pairs/FinQA-MathQA.json", "w"), indent=4)


def sample(path, sample_num, add_nopair=False, neg_samp=False, contra_sub_tree_pos=False, neg_samp_func=RetTrue):
    data1 = json.load(open("../dataset/FinQA_mbert_token_train.json"))["pairs"]
    data2 = json.load(open("../dataset/MathQA_mbert_token_train.json"))["pairs"]

    data = data1 + data2
    pairs = json.load(open(path))

    tree_ls1 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data1]
    tree_ls2 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data2]
    trees = tree_ls1 + tree_ls2
    ops = [tree[0] for tree in trees]
    exprs = [from_infix_to_postfix(_['expression']) for _ in data]
    n_num = [len([_ for _ in expr if _ not in ['+', '-', '*', '/']]) for expr in exprs]

    # total index numbering
    for pair in tqdm.tqdm(pairs):
        if pair[2] == 0:
            pair[1] += len(data1)
        else:
            pair[0] += len(data1)

    # for pair in tqdm.tqdm(pairs):
    #     pair[1] += len(data1)

    # make pairs to dictionary type {'pair[0]':[pair[1]]}, then shuffle
    pairs_dict = dict()
    for pair in tqdm.tqdm(pairs):
        if pair[0] not in pairs_dict:
            pairs_dict[pair[0]] = []
        pairs_dict[pair[0]].append(pair[1])
    random.shuffle(pairs)

    # sample
    d = dict()
    new_pairs = []
    pair_pos = []
    for pair in tqdm.tqdm(pairs):
        if pair[0] not in d:
            d[pair[0]] = 0
        if pair[1] not in d:
            d[pair[1]] = 0
        if d[pair[0]] >= sample_num or d[pair[1]] >= sample_num:
            continue
        d[pair[0]] += 1
        d[pair[1]] += 1
        new_pair = [pair[0], pair[1]]
        new_pairs.append(new_pair)
        print(new_pair)
        if contra_sub_tree_pos:
            pair_pos.append([pair[2], pair[3]])

    # d = [_ for _ in d if d[_] > 0]
    # print(len(d), len(data1) + len(data2))

    if add_nopair:
        tot_len = len(data1) + len(data2)
        for id in range(tot_len):
            if id not in d:
                new_pairs.append([id, id])  # if not match, match it with itself
    if neg_samp:
        for new_pair in tqdm.tqdm(new_pairs):
            for i in range(5):
                cnt = 0
                while (True):
                    if new_pair[0] < len(data1):
                        neg = random.randint(0, len(data2) - 1) + len(data1)
                        if neg not in pairs_dict[new_pair[0]] and neg_samp_func(neg, new_pair[0], ops, n_num):
                            new_pair.append(neg)
                            break
                    else:
                        neg = random.randint(0, len(data1) - 1)
                        if neg not in pairs_dict[new_pair[0]] and neg_samp_func(neg, new_pair[0], ops, n_num):
                            new_pair.append(neg)
                            break
                    cnt += 1
                    if cnt > 1000:
                        new_pair.append(neg)
                        break

    if contra_sub_tree_pos:
        json.dump({"pairs": new_pairs, "pos": pair_pos}, open("../dataset/pairs/FinQA-MathQA-sample.json", "w"), indent=4)
    else:
        json.dump(new_pairs, open("../dataset/pairs/FinQA-MathQA-sample.json", "w"), indent=4)


if __name__ == '__main__':
    main()
    sample('../dataset/pairs/FinQA-MathQA.json', 4, neg_samp=True, contra_sub_tree_pos=True, neg_samp_func=func1)
