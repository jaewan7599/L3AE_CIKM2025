import os, pickle
import numpy as np
from scipy.sparse import coo_matrix
from data_utils.datasets_general_cf import AllRankTstData
import torch.utils.data as data


class DataHandlerLinearCF:
    def __init__(self, configs):
        root_path, self.configs = os.getcwd(), configs
        predir = os.path.join(root_path, f"data/{configs['data']['name']}/")
        self.trn_file, self.val_file, self.tst_file = predir + 'trn_mat.pkl', predir + 'val_mat.pkl', predir + 'tst_mat.pkl'

    def _load_one_mat(self, file):
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)

        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)

        return mat

    def load_data(self):
        trn_mat, val_mat, tst_mat = self._load_one_mat(self.trn_file), self._load_one_mat(self.val_file), self._load_one_mat(self.tst_file)
        self.configs['data']['user_num'], self.configs['data']['item_num'] = trn_mat.shape
        
        val_data, tst_data = AllRankTstData(val_mat, trn_mat), AllRankTstData(tst_mat, trn_mat)
        
        self.trn_mat = trn_mat
        
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
        