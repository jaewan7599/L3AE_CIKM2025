import os, random
from copy import deepcopy
from tqdm import tqdm

import numpy as np
from numpy import random
import torch
import torch.optim as optim

from trainer.metrics import Metric
from models.bulid_model import build_model
from .utils import log_exceptions


def init_seed(configs):
    seed = configs['train']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.mkldnn.enabled = False

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


class Trainer(object):
    def __init__(self, data_handler, configs, logger=None):
        self.configs = configs
        self.data_handler, self.logger, self.metric = data_handler, logger, Metric(self.configs)

    def create_optimizer(self, model):
        optim_config = self.configs['optimizer']
        self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        if self.neg_sampling_per_epoch:
            train_dataloader.dataset.sample_negs(num_negs=self.num_negs)

        loss_log_dict, ep_loss = {}, 0

        model.train()
        for data in train_dataloader:
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(self.configs['device']), data))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if 'log_loss' in self.configs['train'] and self.configs['train']['log_loss']:
            self.logger.log(loss_log_dict, save_to_log=False, print_to_console=True)

    @log_exceptions
    def train(self, model, test_all=True):
        train_config = self.configs['train']
        now_patience, best_epoch, best_valid_recall, best_state_dict, best_valid_result = 0, 0, -1e9, deepcopy(model.state_dict()), None

        self.create_optimizer(model)

        # negative sampling
        if 'num_negs' in train_config:
            self.num_negs = train_config['num_negs']
        else:
            self.num_negs = 1
        
        self.model_name = self.configs['model']['name']
        self.neg_sampling_per_epoch = True

        for epoch_idx in tqdm(range(train_config['epoch']), desc='Training Recommender'):
            self.train_epoch(model, epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result = self.validation(model, epoch_idx)
                
                valid_score = eval_result['recall'][self.configs['test']['k'].index(20)]
                if valid_score > best_valid_recall:
                    now_patience, best_epoch, best_valid_recall, best_state_dict, best_valid_result = 0, epoch_idx, valid_score, deepcopy(model.state_dict()), eval_result
                else:
                    now_patience += 1

                # early stop
                if now_patience == self.configs['train']['patience']:
                    break

        # final test
        model = build_model(self.data_handler, self.configs).to(self.configs['device'])
        model.load_state_dict(best_state_dict)
        test_result = self.test(model, self.data_handler.test_dataloader)

        # save result
        self.save_model(model)
        self.logger.log("Best Epoch {}. Test result: {}.".format(best_epoch, test_result))

        if test_all:
            head_test_result = self.test(model, self.data_handler.test_dataloader, item_filter=self.data_handler.head_items, filter_type="Head")
            tail_test_result = self.test(model, self.data_handler.test_dataloader, item_filter=self.data_handler.tail_items,filter_type="Tail")
            
            unbiased_test_result = self.unbiased_test(model, self.data_handler.test_dataloader, pscore=self.data_handler.pscore)
            
            return best_valid_result, test_result, head_test_result, tail_test_result, unbiased_test_result
        else:
            return best_valid_result, test_result

    @log_exceptions
    def validation(self, model, epoch_idx=None):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
        if self.logger:
            self.logger.log_eval(eval_result, self.configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)

        return eval_result

    @log_exceptions
    def test(self, model, dataloader, item_filter=None, filter_type=None):
        model.eval()
        
        if item_filter is not None:
            eval_result = self.metric.filtered_eval(model, dataloader, item_filter)
            data_type = filter_type + ' items' if filter_type else 'Filtered Test set'
        else:
            eval_result = self.metric.eval(model, dataloader)
            data_type = 'Test set'
        if self.logger:
            self.logger.log_eval(eval_result, self.configs['test']['k'], data_type=data_type)

        return eval_result
    
    @log_exceptions
    def unbiased_test(self, model, dataloader, pscore):
        model.eval()
        eval_result = self.metric.unbiased_eval(model, dataloader, pscore)
        if self.logger:
            self.logger.log_eval(eval_result, self.configs['test']['k'], data_type='Unbiased Test set')

        return eval_result

    @log_exceptions
    def test_save(self, model):
        model.eval()
        eval_result, candidate_set = self.metric.eval_save(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, self.configs['test']['k'], data_type='Test set')

        return eval_result, candidate_set

    def save_model(self, model):
        local_time = self.logger.local_time
        self.saved_model_file = None
        if self.configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = self.configs['model']['name']
            if not self.configs['tune']['enable']:
                save_dir_path = './encoder/checkpoint/{}'.format(model_name)
                self.saved_model_file = '{}/{}-{}-{}.pth'.format(save_dir_path, self.configs['data']['name'], self.configs['train']['seed'], local_time)

                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                torch.save(model_state_dict, self.saved_model_file)
                self.logger.log("Save model parameters to {}".format(self.saved_model_file))
            else:
                save_dir_path = './encoder/checkpoint/{}/tune'.format(model_name)
                now_para_str = self.configs['tune']['now_para_str']
                self.saved_model_file = '{}/{}-{}.pth'.format(save_dir_path, now_para_str, local_time)

                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                torch.save(
                    model_state_dict, self.saved_model_file)
                self.logger.log("Save model parameters to {}".format(self.saved_model_file))

    def load_model(self, model):
        if 'pretrain_path' in self.configs['train']:
            pretrain_path = self.configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))


class AutoCFTrainer(Trainer):
    def __init__(self, data_handler, configs, logger):
        super(AutoCFTrainer, self).__init__(data_handler, configs, logger)

        self.configs = configs
        self.fix_steps = self.configs['model']['fix_steps']

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict, ep_loss = {}, 0
        
        # start this epoch
        model.train()
        for i, data in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(self.configs['device']), data))

            if i % self.fix_steps == 0:
                sampScores, seeds = model.sample_subgraphs()
                encoderAdj, decoderAdj = model.mask_subgraphs(seeds)

            loss, loss_dict = model.cal_loss(batch_data, encoderAdj, decoderAdj)

            if i % self.fix_steps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
                loss_dict['infomax_loss'] = localGlobalLoss

            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        # log loss
        if self.configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)


class LinearTrainer(Trainer):
    def __init__(self, data_handler, configs, logger):
        super(LinearTrainer, self).__init__(data_handler, configs)

        self.configs = configs
    
    @log_exceptions
    def train(self, model, test_all=True):
        valid_result = self.validation(model, 0)
        test_result = self.test(model, self.data_handler.test_dataloader)
                
        if test_all:
            head_test_result = self.test(model, self.data_handler.test_dataloader, item_filter=self.data_handler.head_items, filter_type="Head")
            tail_test_result = self.test(model, self.data_handler.test_dataloader, item_filter=self.data_handler.tail_items, filter_type="Tail")
            
            unbiased_test_result = self.unbiased_test(model, self.data_handler.test_dataloader, pscore=self.data_handler.pscore)

            return valid_result, test_result, head_test_result, tail_test_result, unbiased_test_result
        else:
            return valid_result, test_result



