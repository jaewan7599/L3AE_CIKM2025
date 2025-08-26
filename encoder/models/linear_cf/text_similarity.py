from models.base_linear import BaseLinear
import torch
import numpy as np
import scipy.sparse as sp
from models.text_embedding import load_text_embedding


class Text_Similarity(BaseLinear):
    def __init__(self, data_handler, configs):
        super().__init__(configs)
        
        dataset, llm = configs['data']['name'], configs['data']['llm']
        # X: (# users, # items), F: (# items, # dims)
        X, (_, F) = data_handler.trn_mat, load_text_embedding(dataset, llm, data_handler.trn_mat)

        self.interaction_matrix, self.item_similarity = X.tocsr(), F @ F.T
        np.fill_diagonal(self.item_similarity, 0)

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        user_ids = pck_users.cpu().numpy()

        full_preds = self.interaction_matrix[user_ids, :] @ self.item_similarity
        full_preds = self._mask_predict(torch.from_numpy(full_preds), train_mask)

        return full_preds
