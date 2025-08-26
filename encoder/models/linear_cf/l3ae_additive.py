from models.base_linear import BaseLinear
import torch
import numpy as np
from models.text_embedding import load_text_embedding


class L3AE_Additive(BaseLinear):
    def __init__(self, data_handler, configs):
        super().__init__(configs)

        dataset, llm = configs['data']['name'], configs['llm']
        # X: (# users, # items), F: (# items, # dims)
        X, (_, F) = data_handler.trn_mat, load_text_embedding(dataset, llm, data_handler.trn_mat)
        # F: (# dims, # items)
        F = F.T

        reg_X, reg_F, alpha = self.hyper_config["reg_X"], self.hyper_config["reg_F"], self.hyper_config["alpha"]
        
        GF = F.T @ F
        P2 = np.linalg.inv(GF + reg_F * np.identity(self.item_num))
        C = P2 / -np.diag(P2)
        np.fill_diagonal(C, 0)
        
        GX = X.T @ X
        P = np.linalg.inv(GX + reg_X * np.identity(self.item_num))
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0)
        
        W = (1 - alpha) * B + alpha * C

        self.interaction_matrix, self.item_similarity = X.tocsr(), W

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        user_ids = pck_users.cpu().numpy()

        full_preds = self.interaction_matrix[user_ids, :] @ self.item_similarity
        full_preds = self._mask_predict(torch.from_numpy(full_preds), train_mask)

        return full_preds
