from models.base_linear import BaseLinear
import torch
import numpy as np
from models.text_embedding import load_text_embedding


class L3AE(BaseLinear):
    def __init__(self, data_handler, configs):
        super().__init__(configs)

        dataset, llm = configs['data']['name'], configs['data']['llm']
        # X: (# users, # items), F: (# items, # dims)
        X, (_, F) = data_handler.trn_mat, load_text_embedding(dataset, llm, data_handler.trn_mat)
        # F: (# dims, # items)
        F = F.T
        
        reg_X, reg_F, reg_E = self.hyper_config["reg_X"], self.hyper_config["reg_F"], self.hyper_config["reg_E"]
        
        GF = F.T @ F
        P1 = np.linalg.inv(GF + reg_F * np.identity(self.item_num))
        C = P1 / (-np.diag(P1))
        np.fill_diagonal(C, 0)

        GX = (X.T @ X).todense()
        P2 = np.linalg.inv(GX + (reg_X + reg_E) * np.identity(self.item_num))
        P2C = P2 @ C
        
        B = reg_E * P2C - np.multiply(P2, (1 + reg_E * np.diag(P2C)) / np.diag(P2))
        np.fill_diagonal(B, 0)

        self.interaction_matrix, self.item_similarity = X.tocsr(), B

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        user_ids = pck_users.cpu().numpy()

        full_preds = self.interaction_matrix[user_ids, :] @ self.item_similarity
        full_preds = self._mask_predict(torch.from_numpy(full_preds), train_mask)

        return full_preds
