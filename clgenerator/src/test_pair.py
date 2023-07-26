from pre_data_fin import *
from expressions_transfer import *

import json
import tqdm
import random

def add_order(tree, order):
    if isinstance(tree, list):
        tree.append(order)
        new_order = add_order(tree[1][0], order + 1)
        new_order = add_order(tree[1][1], new_order)
        return new_order
    else:
        return order + 1

def comp(t1, t2):
    if not isinstance(t1, list) and not isinstance(t2, list):
        return True
    if isinstance(t1, list) and isinstance(t2, list) and t1[0] == t2[0]:
        return comp(t1[1][0], t2[1][0]) and comp(t1[1][1], t2[1][1])
    return False

def is_sub_tree(t, s):  # add_order order number return
    if not isinstance(t, list):
        return -1
    if not isinstance(s, list):
        return -1
    if comp(t, s):
        return s[3]
    else:
        x1 = is_sub_tree(t, s[1][0])
        if x1 != -1:
            return x1
        x2 = is_sub_tree(t, s[1][1])
        return x2

def maketree(forword):
    # construct tree
    valid_math_op = ['+', '-', '*', '/', '**', '>', '^']
    valid_table_op = 'T'
    root = None
    now = root
    for word in forword:
        if word in valid_math_op or valid_table_op:
            node = [word, [None, None], None]
            if now == None:
                root = node
                now = root
            else:
                if now[1][0] == None:
                    now[1][0] = node
                    node[2] = now
                    now = node
                else:
                    now[1][1] = node
                    node[2] = now
                    now = node
        else:
            if now == None and root == None:
                root = word
            elif now != None:
                if now[1][0] == None:
                    now[1][0] = word
                else:
                    now[1][1] = word
                    while(now != None and now[1][1] != None):
                        now = now[2]
    if now != None:
        print("error", forword)
    # root = tree_normalize(root)
    return root

def maketree_from_prefix(prefix):
    stack = []
    valid_math_op = ['+', '-', '*', '/', '**', '>', '^', 'T']

    # Read the prefix expression in reverse order
    for char in reversed(prefix):
        if char not in valid_math_op:
            # Create a new node for the operand
            node = Et(char)
            stack.append(node)
        else:
            # Create a new node for the operator
            node = Et(char)
            # Pop two operands from the stack
            node.left = stack.pop()
            node.right = stack.pop()
            stack.append(node)

    # The last node remaining in the stack is the root of the tree
    root = stack.pop()
    return root


def maketree_from_postfix(postfix):
    # print(postfix)
    stack = []
    valid_math_op = ['+', '-', '*', '/', '**', '>', '^']
    valid_table_op = ['T']

    for char in postfix:
        if char not in valid_math_op and valid_table_op:
            # Create a new node for the operand
            node = Et(char)

        elif char in valid_table_op:
            node = Et(char)
            node.right = stack.pop()
            node.left = None

        else:
            # Create a new node for the operator
            node = Et(char)
            # Pop two operands from the stack
            node.right = stack.pop()
            node.left = stack.pop()

        stack.append(node)

    # The last node remaining in the stack is the root of the tree
    root = stack.pop()
    return root


# def sample(path, sample_num, add_nopair=False, neg_samp=False, contra_sub_tree_pos=False, neg_samp_func=RetTrue):
data1 = json.load(open("../dataset/FinQA_mbert_token_train.json"))["pairs"]
# data2 = json.load(open("../dataset/MathQA_mbert_token_train.json"))["pairs"]

# data = data1 # + data2
# pairs = json.load(open(path))

for _ in data1:
    print(_['expression'])
    exp = infix_to_postfix(_['expression'])
    print(maketree_from_postfix(exp))

tree_ls1 = [maketree_from_postfix(infix_to_postfix(_['expression'])) for _ in data1]
# tree_ls2 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data2]

for _ in tree_ls1:
    add_order(_, 0)
    print(len(_))

# trees = tree_ls1 + tree_ls2
ops = [tree[0] for tree in tree_ls1]
exprs = [infix_to_postfix(_['expression']) for _ in data1]
n_num = [len([_ for _ in expr if _ not in ['+', '-', '*', '/', '**', '>', '^', 'T']]) for expr in exprs]

for pair in tqdm.tqdm(pairs):
    if len(pair) < 3 or pair[2] == 0:
        pair[1] += len(data1)
    else:
        pair[0] += len(data1)

    # for pair in tqdm.tqdm(pairs):
    #     pair[1] += len(data1)

pairs_dict = dict()
for pair in tqdm.tqdm(pairs):
    if pair[0] not in pairs_dict:
        pairs_dict[pair[0]] = []
    pairs_dict[pair[0]].append(pair[1])
random.shuffle(pairs)

print(pairs)
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
    print(new_pair)
    new_pairs.append(new_pair)
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

