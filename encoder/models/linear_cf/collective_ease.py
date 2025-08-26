from models.base_linear import BaseLinear
import torch
import numpy as np
import scipy.sparse as sp
from models.text_embedding import load_text_embedding


class Collective_EASE(BaseLinear):
    def __init__(self, data_handler, configs):
        super().__init__(configs)

        dataset, llm = configs['data']['name'], configs['data']['llm']
        # X: (# users, # items), F: (# items, # words)
        X, (_, F) = data_handler.trn_mat, load_text_embedding(dataset, llm, data_handler.trn_mat, is_dense=False)
        # F: (# words, # items)
        F = F.T.astype(np.float32)
        reg, alpha = self.hyper_config["reg"], self.hyper_config["alpha"]

        F *= alpha
        XF = sp.vstack([X, F])
        
        G = (XF.T @ XF).toarray()
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
