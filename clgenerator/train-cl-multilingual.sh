#!/bin/bash

python3 run_cl.py \
    --output_dir output/mwp-cl-multilingual-fin \
    --bert_pretrain_path bert-base-uncased \
    --data_dir dataset \
    --train_file FinQA-FinQA_mbert_token_train.json \
    --dev_file_1 FinQA_mbert_token_val.json \
    --test_file_1 FinQA_mbert_token_test.json \
    --dev_file_2 FinQA_mbert_token_val.json \
    --test_file_2 FinQA_mbert_token_test.json \
    --contra_pair pairs/FinQA-FinQA-sample1.json \
    --learning_rate 5e-5 \
    --n_epochs 22 \
    --n_val 1 \
    --hidden_size 768 \
    --batch_size 16 \
    --embedding_size 128 \
    --beam_size 3 \
    --contra_loss_func margin \
    --contra_common_tree_pair \
    --contra_loss_margin 0.15 \
    --neg_sample 1 \
    --neg_sample_from_pair_file \
    --neg_no_expr_loss \
    --alpha 5 \
    --alpha_warmup \
    --warmup_begin 1000 \
    --warmup_end 6000 \
    --dropout 0.5 \
    --seed 16
