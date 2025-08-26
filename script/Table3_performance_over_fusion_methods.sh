#!/bin/bash
dataset=games
python encoder/train_linear.py --dataset $dataset --device cpu --llm nv-embed-v2 --model l3ae_collective --reg 100 --alpha 5
python encoder/train_linear.py --dataset $dataset --device cpu --llm nv-embed-v2 --model l3ae_additive --reg_X 100 --reg_F 10 --alpha 0.6
python encoder/train_linear.py --dataset=$dataset --device cpu --llm nv-embed-v2 --model l3ae --reg_X -50 --reg_F 5 --reg_E 150

dataset=toys
python encoder/train_linear.py --dataset $dataset --device cpu --llm nv-embed-v2 --model l3ae_collective --reg 500 --alpha 4
python encoder/train_linear.py --dataset $dataset --device cpu --llm nv-embed-v2 --model l3ae_additive --reg_X 100 --reg_F 5 --alpha 0.6
python encoder/train_linear.py --dataset=$dataset --device cpu --llm nv-embed-v2 --model l3ae --reg_X -100 --reg_F 10 --reg_E 200

dataset=books
python encoder/train_linear.py --dataset $dataset --device cpu --llm nv-embed-v2 --model l3ae_collective --reg 100 --alpha 5
python encoder/train_linear.py --dataset $dataset --device cpu --llm nv-embed-v2 --model l3ae_additive --reg_X 50 --reg_F 1 --alpha 0.4
python encoder/train_linear.py --dataset=$dataset --device cpu --llm nv-embed-v2 --model l3ae --reg_X -100 --reg_F 10 --reg_E 150
