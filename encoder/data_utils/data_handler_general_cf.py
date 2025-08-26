import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_general_cf import PairwiseTrnData, PairwiseWEpochFlagTrnData, AllRankTstData
import torch as t
import torch.utils.data as data


class DataHandlerGeneralCF:
    def __init__(self, configs):
        root_path, self.configs = os.getcwd(), configs
        predir = os.path.join(root_path, f'data/{configs['data']['name']}/')
        self.trn_file, self.val_file, self.tst_file = predir + 'trn_mat.pkl', predir + 'val_mat.pkl', predir + 'tst_mat.pkl'

    def _load_one_mat(self, file):
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)

        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)

        return mat

    def _normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)

        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat, self_loop=False):
        num_users, num_items = self.configs['data']['user_num'], self.configs['data']['item_num']
        if not self_loop:
            a, b = csr_matrix((num_users, num_users)), csr_matrix((num_items, num_items))
        else:
            data, row_indices, column_indices = np.ones(num_users), np.arange(num_users), np.arange(num_users)
            a = csr_matrix((data, (row_indices, column_indices)), shape=(num_users, num_users))

            data, row_indices, column_indices = np.ones(num_items), np.arange(num_items), np.arange(num_items)
            b = csr_matrix((data, (row_indices, column_indices)), shape=(num_items, num_items))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs, vals, shape = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64)), t.from_numpy(mat.data.astype(np.float32)), t.Size(mat.shape)
        
        return t.sparse.FloatTensor(idxs, vals, shape).to(self.configs['device'])

    def load_data(self):
        trn_mat, val_mat, tst_mat = self._load_one_mat(self.trn_file), self._load_one_mat(self.val_file), self._load_one_mat(self.tst_file)
        self.configs['data']['user_num'], self.configs['data']['item_num'] = trn_mat.shape
        
        if self.configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat, self.configs)
        elif self.configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat, self.configs)

        val_data, tst_data = AllRankTstData(val_mat, trn_mat), AllRankTstData(tst_mat, trn_mat)
        
        self.trn_mat = trn_mat
        if self.configs['model']['name'] == 'gccf':
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)
        else:
            self.torch_adj = self._make_torch_adj(trn_mat)
        
        self.train_dataloader = data.DataLoader(trn_data, batch_size=self.configs['train']['batch_size'], shuffle=True,
                                                num_workers=0)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=self.configs['test']['batch_size'], shuffle=False,
                                                num_workers=0)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=self.configs['test']['batch_size'], shuffle=False,
                                               num_workers=0)
        
        self.item_popularity = np.array(trn_mat.sum(axis=0)).flatten()
        
        pscore = (self.item_popularity / self.item_popularity.max()) ** 0.5
        self.pscore = pscore.clip(min=1e-3)

        item_ranks = np.argsort(self.item_popularity)[::-1]

        num_items = trn_mat.shape[1]
        head_size = int(num_items * 0.2)
        self.head_items, self.tail_items = item_ranks[:head_size], item_ranks[head_size:]

        