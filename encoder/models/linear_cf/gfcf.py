from models.base_linear import BaseLinear
import torch
import numpy as np
import scipy.sparse as sp


class GFCF(BaseLinear):
    def __init__(self, data_handler, configs):
        super().__init__(configs)
        
        X = data_handler.trn_mat.tolil()

        rowsum = np.array(X.sum(axis=1))
        D_inv = np.power(rowsum, -0.5).flatten()
        D_inv[np.isinf(D_inv)] = 0.
        D_mat = sp.diags(D_inv)
        normalized_X = D_mat.dot(X)

        colsum = np.array(X.sum(axis=0))
        D_inv = np.power(colsum, -0.5).flatten()
        D_inv[np.isinf(D_inv)] = 0.
        D = sp.diags(D_inv)
        normalized_X = normalized_X.dot(D)

        self.D, self.D_inv, self.normalized_X = D, sp.diags(1/D_inv), normalized_X.tocsc()
        _, _, self.VT = sp.linalg.svds(self.normalized_X, 256)
        self.normed_G, self.truncated_G = self.normalized_X.T @ self.normalized_X, self.D @ self.VT.T @ self.VT @ self.D_inv
        self.interaction_matrix = X

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        user_ids = pck_users.cpu().numpy()

        batch_test = np.array(self.interaction_matrix[user_ids,:].todense())
        full_preds2 = batch_test @ self.normed_G
        full_preds1 = batch_test @ self.truncated_G

        full_preds = full_preds2 + 0.3 * full_preds1
        full_preds = self._mask_predict(torch.from_numpy(full_preds), train_mask)

        return full_preds
