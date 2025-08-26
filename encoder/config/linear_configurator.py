import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn


def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    # model and dataset
    parser.add_argument('--model', type=str, default='LightGCN', help='model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='dataset name')
    parser.add_argument('--llm', type=str, help='language model for embedding') # ['nv-embed-v2', 'llama-3.2-3b', 'qwen-embedding-8b']

    # hyperparameters
    parser.add_argument('--save_result_file', action='store_true', help='save the result file')
    
    # seed and device
    parser.add_argument('--seed', type=int, default=2023, help='seed number')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--cuda', type=str, default='0', help='device number')

    # EASE, Additive_EASE, L3AE
    parser.add_argument('--reg_X', type=float, default=1, help='L2 Regularization for X')

    # EASE_Text, Additive_EASE, L3AE
    parser.add_argument('--reg_F', type=float, default=1, help='L2 Regularization for F')

    # Collective_EASE
    parser.add_argument('--reg', type=float, default=1, help='L2 Regularization for XF')
    parser.add_argument('--alpha', type=float, default=1, help='Weight for F')

    # L3AE
    parser.add_argument('--reg_E', type=float, default=1, help='Weight for Distillation')

    # SGFCF
    parser.add_argument('--mu', type=float, default=0, help='')
    parser.add_argument('--eps', type=float, default=0.5, help='')
    parser.add_argument('--k', type=float, default=100, help='')
    parser.add_argument('--beta', type=float, default=0.5, help='')
    parser.add_argument('--beta_end', type=float, default=2, help='')
    parser.add_argument('--gamma', type=float, default=1.6, help='')
    parser.add_argument('--use_igf', action='store_true', help='')

    # BSPM
    parser.add_argument('--merge', type=str, default='EM', help='EM, LM')
    parser.add_argument('--solver_shr', type=str, default='rk4', help='rk4, euler')
    parser.add_argument('--K_s', type=int, default=1, help='K_s: {1, 2, 3, 4}')
    parser.add_argument('--T_s', type=float, default=1.2, help='T_s: [1, 5]')
    parser.add_argument('--t_point_combination', type=eval, default=True, help='True, False')
    parser.add_argument('--factor_dim', type=int, default=256, help='factor_dim: [256, 384, 448]')

    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'

    # dataset
    if dataset is not None:
        args.dataset = dataset

    root_path = os.getcwd()

    # find yml file
    if not os.path.exists(os.path.join(root_path, 'encoder/config/modelconf/{}.yml'.format(model_name))):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open(os.path.join(root_path, 'encoder/config/modelconf/{}.yml'.format(model_name)), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        configs['data']['name'] = args.dataset
        configs['device'] = args.device
        configs['train']['seed'] = args.seed
        configs['test']['save_result_file'] = args.save_result_file

        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
            
        if args.llm:
            configs['data']['llm'] = args.llm
        if 'llm' not in configs['data']:
            configs['data']['llm'] = None

        configs['optimizer'] = None

        for params in ['reg_X', 'reg_F', 'reg_E', 'alpha', 'reg',
                       'mu', 'eps', 'k', 'beta', 'beta_end', 'gamma', 'use_igf',
                       'num_hops', 'merge', 'solver_shr', 'K_s', 'T_s', 't_point_combination', 'factor_dim']:
            if params in configs['model']:
                configs['model'][params] = getattr(args, params)
                
        return configs
    

configs = parse_configure()
