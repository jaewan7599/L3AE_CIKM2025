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

    # seed and device
    parser.add_argument('--seed', type=int, default=2023, help='seed number')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--cuda', type=str, default='0', help='device number')

    # hyperparameters
    parser.add_argument('--save_result_file', action='store_true', help='save the result file')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size for model training')
    parser.add_argument('--epoch', type=int, default=3000, help='max epoch of training')
    parser.add_argument('--emb_size', type=int, default=32, help='embedding size of a model')
    parser.add_argument('--num_negs', type=int, default=1, help='the number of negative samples for each positive interaction')
    parser.add_argument('--reg_weight', type=float, default=1.0e-6, help='regularization weight for the model')
    parser.add_argument('--lr', type=float, default=1.0e-3, help='learning rate of optimizer')
    parser.add_argument('--llm', type=str, help='language model for embedding') # ['nv-embed-v2', 'llama-3.2-3b', 'qwen-embedding-8b']

    # LightGCN
    parser.add_argument('--layer_num', type=int, default=2, help='the number of GCN layers')

    # SimGCL
    parser.add_argument('--cl_tau', type=float, default=0.15, help='temperature of InfoNCE loss')
    parser.add_argument('--cl_weight', type=float, default=0.1, help='weight of InfoNCE loss')
    parser.add_argument('--eps', type=float, default=0.1, help='')

    # RLMRec-Con (Constrative Alignment)
    parser.add_argument('--kd_tau', type=float, default=0.15, help='temperature of KD loss')
    parser.add_argument('--kd_weight', type=float, default=0.1, help='weight of KD loss')

    # RLMRec-Gen
    parser.add_argument('--mask_ratio', type=float, default=0.1, help='mask ratio for RLMRec-Gen')
    parser.add_argument('--recon_weight', type=float, default=1.0e-1, help='weight of reconstruction loss')
    parser.add_argument('--recon_tau', type=float, default=0.2, help='temperature of reconstruction loss')
    
    # AlphaRec
    parser.add_argument('--no_pred_norm', action='store_true', help='whether to normalize the prediction scores')
        
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
        configs['model']['embedding_size'] = args.emb_size

        configs['data']['name'] = args.dataset
        if args.llm:
            configs['data']['llm'] = args.llm
        if 'llm' not in configs['data']:
            configs['data']['llm'] = None

        if 'tune' not in configs:
            configs['tune'] = {'enable': False}


        configs['device'] = args.device

        configs['train']['epoch'] = args.epoch
        configs['train']['seed'] = args.seed
        configs['train']['batch_size'] = args.batch_size
        configs['train']['num_negs'] = args.num_negs

        configs['test']['save_result_file'] = args.save_result_file

        for params in ['reg_weight', 'layer_num', # LightGCN
                       'cl_weight', 'cl_tau', 'eps', # SimGCL
                       'kd_weight', 'kd_tau', # Contrastive Alignment
                       'mask_ratio', 'recon_weight', 'recon_tau', # Reconstruction
                       'no_pred_norm']: # AlphaRec
            if params in configs['model']:
                configs['model'][params] = getattr(args, params)

        return configs

configs = parse_configure()
