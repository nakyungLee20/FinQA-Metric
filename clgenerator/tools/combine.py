import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


with open('../dataset/FinQA_mbert_token_train.json') as f1:
	data1 = json.load(f1)

with open('../dataset/FinQA_mbert_token_train.json') as f2:
	data2 = json.load(f2)

data = dict()
data['pairs']=data1['pairs']+data2['pairs']
data['generate_nums']=data1['generate_nums']+data2['generate_nums']
data['copy_nums']=data1['copy_nums']+data2['copy_nums']

with open("../dataset/FinQA-FinQA_mbert_token_train.json", "w") as nf:
	json.dump(data, nf)

print(len(data['pairs']))
print(len(data1['pairs'])+len(data2['pairs']))