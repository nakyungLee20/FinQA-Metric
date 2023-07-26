#!/bin/bash

python3 run_ft.py \
    --output_dir output/mwp-ft-multilingual-en-fin \
    --bert_pretrain_path bert-base-uncased \
    --model_reload_path output/mwp-cl-multilingual/epoch_15 \
    --data_dir dataset \
    --finetune_from_trainset FinQA-FinQA_mbert_token_train.json \
    --train_file FinQA-FinQA_mbert_token_train.json \
    --dev_file FinQA_mbert_token_val.json \
    --test_file FinQA_mbert_token_test.json \
    --n_val 1 --n_save_ckpt 10 --schedule linear --batch_size 16 --learning_rate 5e-5 --n_epochs 50 \
    --warmup_steps 4500 --hidden_size 768 --beam_size 3 --dropout 0.5 --seed 17