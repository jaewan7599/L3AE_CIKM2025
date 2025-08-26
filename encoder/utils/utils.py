import os


def get_linear_model_header(metrics, top_k, model, save_all):
    if model == 'sgfcf':
        header = "train_time,mu,eps,k,beta,beta_end,gamma,use_igf,"
    elif model == 'bspm':
        header = "train_time,merge,solver_shr,K_s,T_s,t_point_combination,beta,factor_dim,"
    else:
        header = "train_time,reg,reg_X,reg_F,reg_E,alpha,beta,num_hops,"
    
    if save_all:
        mode_list = ['valid', 'test', 'head', 'tail', 'unbiased']
    else:
        mode_list = ['valid', 'test']    
    
    for mode in mode_list:
        for metric in metrics:
            for k in top_k:
                header += f"{mode}_{metric}@{k},"
    header += "\n"

    return header


def get_linear_model_parameter(hyper_config, model):
    res = ""
    if model == 'sgfcf':
        res += (
            f"{hyper_config['mu']},"
            f"{hyper_config['eps']},"
            f"{hyper_config['k']},"
            f"{hyper_config['beta']},"
            f"{hyper_config['beta_end']},"
            f"{hyper_config['gamma']},"
            f"{hyper_config['use_igf']},"
        )
        return res
    elif model == 'bspm':
        res += (
            f"{hyper_config['merge']},"
            f"{hyper_config['solver_shr']},"
            f"{hyper_config['K_s']},"
            f"{hyper_config['T_s']},"
            f"{hyper_config['t_point_combination']},"
            f"{hyper_config['beta']},"
            f"{hyper_config['factor_dim']},"
        )
    else:
        for hyper in ['reg', 'reg_X', 'reg_F', 'reg_E', 'alpha',
                      'beta', 'num_hops']:
            if hyper in hyper_config:
                res += f"{hyper_config[hyper]},"
            else:
                res += ","
        
    return res


def save_linear_result(save_all, configs, train_time, valid_result, test_result, head_result=None, tail_result=None, unbiased_result=None):
    dataset, llm, model = configs['data']['name'], configs['data']['llm'], configs['model']['name']
    metrics, top_k = configs['test']['metrics'], configs['test']['k']

    root_path = os.getcwd()

    save_prefix = os.path.join(root_path, f'saves_{llm}', dataset, 'linear')
    os.makedirs(save_prefix, exist_ok=True)
    save_path = os.path.join(save_prefix, f"{model}.csv")
    res = ""
    
    if not os.path.exists(save_path):
        res += get_linear_model_header(metrics, top_k, model, save_all)

    res += f"{train_time},"
    res += get_linear_model_parameter(configs['model'], model)

    for metric in metrics:
        res += f"{','.join(valid_result[metric].astype(str))},"
    for metric in metrics:
        res += f"{','.join(test_result[metric].astype(str))},"
        
    if save_all:
        for metric in metrics:
            res += f"{','.join(head_result[metric].astype(str))},"
        for metric in metrics:
            res += f"{','.join(tail_result[metric].astype(str))},"
        for metric in metrics:
            res += f"{','.join(unbiased_result[metric].astype(str))},"

    res += "\n"
    with open(save_path, 'a+') as f:
        f.write(res)


def get_neural_model_header(metrics, top_k, model, save_all):
    header = "train_time,log_path,reg_weight,layer_num,cl_weight,cl_tau,eps,kd_weight,kd_tau,mask_ratio,recon_weight,recon_tau,no_pred_norm,"
    
    if save_all:
        mode_list = ['valid', 'test', 'head', 'tail', 'unbiased']
    else:
        mode_list = ['valid', 'test']  
        
    for mode in mode_list:
        for metric in metrics:
            for k in top_k:
                header += f"{mode}_{metric}@{k},"
    header += "\n"

    return header


def get_neural_model_parameter(hyper_config, model):
    res = ""
    for hyper in ['reg_weight', 'layer_num', 'cl_weight', 'cl_tau', 'eps', 'kd_weight', 'kd_tau', 'mask_ratio', 'recon_weight', 'recon_tau', 'no_pred_norm']:
        if hyper in hyper_config:
            res += f"{hyper_config[hyper]},"
        else:
            res += ","
    
    return res


def save_neural_result(save_all, configs, train_time, log_file_path, valid_result, test_result, head_result=None, tail_result=None, unbiased_result=None):
    dataset, llm, model = configs['data']['name'], configs['data']['llm'], configs['model']['name']
    metrics, top_k = configs['test']['metrics'], configs['test']['k']

    root_path = os.getcwd()

    save_prefix = os.path.join(root_path, f'saves_{llm}', dataset, 'neural')
    os.makedirs(save_prefix, exist_ok=True)

    save_path = os.path.join(save_prefix, f"{model}.csv")
    res = ""
    
    if not os.path.exists(save_path):
        res += get_neural_model_header(metrics, top_k, model, save_all)

    res += f"{train_time},{log_file_path},"
    res += get_neural_model_parameter(configs['model'], model)

    for metric in metrics:
        res += f"{','.join(valid_result[metric].astype(str))},"
    for metric in metrics:
        res += f"{','.join(test_result[metric].astype(str))},"
        
    if save_all: 
        for metric in metrics:
            res += f"{','.join(head_result[metric].astype(str))},"
        for metric in metrics:
            res += f"{','.join(tail_result[metric].astype(str))},"
        for metric in metrics:
            res += f"{','.join(unbiased_result[metric].astype(str))},"

    res += "\n"
    with open(save_path, 'a+') as f:
        f.write(res)
