from models.base_linear import BaseLinear
import torch
import numpy as np
from models.text_embedding import load_text_embedding


class L3AE_Collective(BaseLinear):
    def __init__(self, data_handler, configs):
        super().__init__(configs)

        dataset, llm = configs['data']['name'], configs['data']['llm']
        # X: (# users, # items), F: (# items, # dims)
        X, (_, F) = data_handler.trn_mat, load_text_embedding(dataset, llm, data_handler.trn_mat)
        # F: (# dims, # items)
        F = F.T
        
        reg, alpha = self.hyper_config["reg"], self.hyper_config["alpha"]

        F *= alpha
        XF = np.vstack([X.toarray(), F])
        
        G = XF.T @ XF
        P = np.linalg.inv(G + reg * np.identity(self.item_num))
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0)

        self.interaction_matrix, self.item_similarity = X.tocsr(), B

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        user_ids = pck_users.cpu().numpy()

        full_preds = self.interaction_matrix[user_ids, :] @ self.item_similarity
        full_preds = self._mask_predict(torch.from_numpy(full_preds), train_mask)

        return full_preds
