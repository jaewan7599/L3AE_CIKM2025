#!/bin/bash
dataset=toys; gpu=0

### non-linear
python encoder/train_encoder.py --dataset $dataset --cuda $gpu --model lightgcn --reg_weight 1e-8
python encoder/train_encoder.py --dataset $dataset --cuda $gpu --model simgcl --reg_weight 1e-8 --cl_weight 0.02 --cl_tau 0.15 --eps 0.1

python encoder/train_encoder.py --dataset $dataset --cuda $gpu --llm qwen3-embedding-8b --model simgcl_plus --reg_weight 0 --cl_weight 0.1 --cl_tau 0.4 --kd_weight 0.02 --kd_tau 0.2
python encoder/train_encoder.py --dataset $dataset --cuda $gpu --llm qwen3-embedding-8b --model simgcl_gene --reg_weight 0 --cl_weight 0.02 --cl_tau 0.15 --mask_ratio 0.1 --recon_weight 0.05 --recon_tau 0.1
python encoder/train_encoder.py --dataset $dataset --cuda $gpu --llm qwen3-embedding-8b --model alpharec --reg_weight 1e-6 --cl_tau 0.15 --no_pred_norm

### linear
python encoder/train_linear.py --dataset $dataset --device cpu --llm qwen3-embedding-8b --model text_similarity
python encoder/train_linear.py --dataset $dataset --device cpu --llm qwen3-embedding-8b --model ease_text --reg_F 0.5

python encoder/train_linear.py --dataset $dataset --device cpu --model ease --reg_X 100
python encoder/train_linear.py --dataset $dataset --device cpu --model gfcf
python encoder/train_linear.py --dataset $dataset --device cpu --model bspm --K_s 2 --T_s 2.2 --t_point_combination True --factor_dim 256
python encoder/train_linear.py --dataset $dataset --device cpu --model sgfcf --eps 0.4 --k 200 --beta 0.4 --beta_end 2 --use_igf True

python encoder/train_linear.py --dataset $dataset --device cpu --model collective_ease --reg 100 --alpha 0.5
python encoder/train_linear.py --dataset $dataset --device cpu --model additive_ease --reg_X 50 --reg_F 100 --alpha 0.4

python encoder/train_linear.py --dataset=$dataset --device cpu --llm qwen3-embedding-8b --model l3ae --reg_X -50 --reg_F 10 --reg_E 150
