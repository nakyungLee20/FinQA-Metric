import pandas as pd
import numpy as np
import json
import copy

f1 = open("train_pairs.json", encoding="utf-8")
train_pairs = json.load(f1)
# print(len(train_pairs))

f2 = open("pairs/FinQA-FinQA-sample2.json", encoding="utf-8")
samples = json.load(f2)
contra_pair = copy.deepcopy(samples['pairs'])
subtree_pos_pair = copy.deepcopy(samples['pos'])

contra_pair_item_len = len(contra_pair[0])
# print(contra_pair_item_len)
merge = [ x + y for x, y in zip(contra_pair, subtree_pos_pair) if max(y) < 45]
#print(merge)
# print(len(merge))

subtree_pos_pair = [_[contra_pair_item_len:] for _ in merge]
# print(subtree_pos_pair)

pos = 0
subtree_pos_pair_batches = []
while pos + 16 < len(subtree_pos_pair):
    subtree_pos_pair_batches.append(subtree_pos_pair[pos:pos+16])
    pos += 16
subtree_pos_pair_batches.append(subtree_pos_pair[pos:])
# print(len(subtree_pos_pair_batches[20][1]))
print(subtree_pos_pair_batches[183][7])  # subtree_pos_pair[i]
print(subtree_pos_pair_batches[183][7])