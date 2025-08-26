#!/bin/bash
dataset=games; gpu=0

### non-linear
python encoder/train_encoder.py --dataset $dataset --cuda $gpu --model lightgcn --reg_weight 1e-6
python encoder/train_encoder.py --dataset $dataset --cuda $gpu --model simgcl --reg_weight 1e-6 --cl_weight 0.02 --cl_tau 0.2 --eps 0.01

python encoder/train_encoder.py --dataset $dataset --cuda $gpu --llm nv-embed-v2 --model simgcl_plus --reg_weight 1e-6 --cl_weight 0.02 --cl_tau 0.2 --kd_weight 0.01 --kd_tau 0.2
python encoder/train_encoder.py --dataset $dataset --cuda $gpu --llm nv-embed-v2 --model simgcl_gene --reg_weight 1e-6 --cl_weight 0.02 --cl_tau 0.2 --mask_ratio 0.1 --recon_weight 0.1 --recon_tau 0.1
python encoder/train_encoder.py --dataset $dataset --cuda $gpu --llm nv-embed-v2 --model alpharec --reg_weight 1e-6 --cl_tau 0.25 --no_pred_norm

### linear
python encoder/train_linear.py --dataset $dataset --device cpu --llm nv-embed-v2 --model text_similarity
python encoder/train_linear.py --dataset $dataset --device cpu --llm nv-embed-v2 --model ease_text --reg_F 10

python encoder/train_linear.py --dataset $dataset --device cpu --model ease --reg_X 100
python encoder/train_linear.py --dataset $dataset --device cpu --model gfcf
python encoder/train_linear.py --dataset $dataset --device cpu --model bspm --K_s 1 --T_s 1.2 --t_point_combination True --factor_dim 256
python encoder/train_linear.py --dataset $dataset --device cpu --model sgfcf --eps 0.4 --k 100 --beta 0.5 --beta_end 1.5 --use_igf True

python encoder/train_linear.py --dataset $dataset --device cpu --model collective_ease --reg 100 --alpha 0.1
python encoder/train_linear.py --dataset $dataset --device cpu --model additive_ease --reg_X 100 --reg_F 50 --alpha 0.2

python encoder/train_linear.py --dataset=$dataset --device cpu --llm nv-embed-v2 --model l3ae --reg_X -50 --reg_F 5 --reg_E 150