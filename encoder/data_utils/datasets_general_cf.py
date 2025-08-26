from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
import numpy as np
from tqdm import tqdm


class PairwiseTrnData(data.Dataset):
    def __init__(self, coomat, configs):
        self.rows, self.cols, self.dokmat = coomat.row, coomat.col, coomat.todok()
        self.negs = np.zeros(len(self.rows), dtype=np.int32)
        self.num_items = configs['data']['item_num']

    def sample_negs(self, num_negs=1):
        if num_negs == 1:
            self.negs = np.zeros(len(self.rows), dtype=np.int32)
            for i in range(len(self.rows)):
                user_id = self.rows[i]
                self.negs[i] = self.sample_user_negs(user_id)
        else:
            self.negs = np.zeros((len(self.rows), num_negs), dtype=np.int32)
            for i in range(len(self.rows)):
                user_id = self.rows[i]
                for j in range(num_negs):
                    self.negs[i, j] = self.sample_user_negs(user_id)
    
    def sample_user_negs(self, user_id):
        while True:
            iNeg = np.random.randint(self.num_items)
            if (user_id, iNeg) not in self.dokmat:
                return iNeg
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class PairwiseWEpochFlagTrnData(PairwiseTrnData):
    def __init__(self, coomat, configs):
        super(PairwiseWEpochFlagTrnData, self).__init__(coomat, configs)
        self.epoch_flag_counter, self.epoch_period = -1, configs['model']['epoch_period']

    def __getitem__(self, idx):
        flag = 0
        if self.epoch_flag_counter == -1:
            flag, self.epoch_flag_counter = 1, 0
        if idx == 0:
            self.epoch_flag_counter += 1
            if self.epoch_flag_counter % self.epoch_period == 0:
                flag = 1
        anc, pos, neg = super(PairwiseWEpochFlagTrnData, self).__getitem__(idx)

        return anc, pos, neg, flag


class AllRankTstData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0
        user_pos_lists, test_users = [list() for _ in range(coomat.shape[0])], set()

        for i in range(len(coomat.data)):
            row, col = coomat.row[i], coomat.col[i]

            user_pos_lists[row].append(col)
            test_users.add(row)

        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists
    
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])

        return pck_user, pck_mask
    
