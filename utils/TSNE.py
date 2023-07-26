from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from clgenerator.src.pre_data_fin import *
from clgenerator.src.models import *
from transformers.models.bert import BertTokenizer, BertModel
import torch

def mbert_tokenize():
    data = load_fin_data("../clgenerator/dataset/train.json")
    pairs, generate_nums, copy_nums = transfer_num(data)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        # print(new_item['expression'])
        new_items.append(new_item)

    return new_items

# data filetering to float type
data = load_fin_data('../clgenerator/dataset/train.json')
data_df = pd.DataFrame(data)
# mdata = mbert_tokenize() # list type

# embedding model and tsne model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# emb_model = BertModel.from_pretrained("bert-base-uncased")  # model epoch best performance call
encoder = EncoderBert(hidden_size=768, auto_transformer=False, bert_pretrain_path="bert-base-uncased", dropout=0.5)
encoder.load_state_dict(torch.load(os.path.join("../clgenerator/output/mwp-ft-multilingual-en", "epoch_12", "encoder.ckpt")))
tsne_model = TSNE(n_components = 2)

# tsne_data = []

for eq in data_df['original_text']:
    #rm_str = eq[3:-1]
    input_id = tokenizer(eq, return_tensors="pt")['input_ids']
    attention = tokenizer(eq, return_tensors="pt")['attention_mask']
    out1, out2 = encoder(input_ids=input_id, attention_mask=attention)
    print(out1.shape) # [11,1,768]
    out = out1.squeeze().detach().numpy()
    print(out[0])

    for layer in range(len(out[0])):
        tsne

    #for layer in range(len(outputs['hidden_states'])):
    #    out = outputs['hidden_states'][layer].squeeze()
    #    out = out.detach().numpy()
    #    tsne_data.append(out)

colors=["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E", "#336680", "#ff7f0e", "#9467bd" ]
tsne_emb =[]

for d in range(int(len(tsne_data)/13)):
  print(d)
  for layer in range(13):
    #print(tsne_data[13*d+layer].shape[0])
    embedding = tsne_model.fit_transform(tsne_data[13*d+layer].T)
    plt.scatter(embedding[:,0].mean(), embedding[:,1].mean(), color=colors[layer])

plt.xlabel('t-SNE dim 0')
plt.ylabel('t-SNE dim 1')
plt.legend()
plt.show()

# encoder = EncoderBert(768, auto_transformer=False, bert_pretrain_path='bert-base-uncased', dropout=0.5)
# tsne_data = encoder.visualize_data()

#for eq in data_df['equation']:
#    rm_str = eq[3:-1]
#    input_id = tokenizer(rm_str, return_tensors="pt")['input_ids']
#    attention = tokenizer(rm_str, return_tensors="pt")['attention_mask']
#    outputs = emb_model(input_ids=input_id, attention_mask=attention, output_hidden_states=True, return_dict=True)
#    for layer in range(len(outputs['hidden_states'])):
#        out = outputs['hidden_states'][layer].squeeze().T
#        out = out.detach().numpy()  # (768,15) shape
#        embedding = tsne_model.fit_transform(out)  # (768,2) shape

#        plt.scatter(embedding[:, 0].mean(), embedding[:, 1].mean(), label=layer)

# tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
# data_df['target'] = df['target']

