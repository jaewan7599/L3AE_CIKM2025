from models.base_linear import BaseLinear
import torch
import numpy as np
import scipy.sparse as sp


class EASE(BaseLinear):
    def __init__(self, data_handler, configs):
        super().__init__(configs)

        X = data_handler.trn_mat
        
        reg_X = self.hyper_config["reg_X"]

        G = X.T @ X
        G += reg_X * sp.identity(self.item_num).astype(np.float32)
        G = G.todense()

        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0)

        self.interaction_matrix, self.item_similarity = X.tocsr(), B

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        user_ids = pck_users.cpu().numpy()

        full_preds = self.interaction_matrix[user_ids, :] @ self.item_similarity
        full_preds = self._mask_predict(torch.from_numpy(full_preds), train_mask)

        return full_preds
