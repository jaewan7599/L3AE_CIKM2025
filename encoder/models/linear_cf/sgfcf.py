from models.base_linear import BaseLinear
import torch
import numpy as np
import scipy.sparse as sp


class SGFCF(BaseLinear):
    def __init__(self, data_handler, configs):
        super().__init__(configs)

        X = data_handler.trn_mat.tocsr()
        num_users, num_items = X.shape
        item_counts, user_counts = X.sum(axis=0), X.sum(axis=1)

        mu, eps, k = self.hyper_config["mu"], self.hyper_config["eps"], int(self.hyper_config["k"])
        beta, beta_end, gamma, use_igf = self.hyper_config["beta"], self.hyper_config["beta_end"], self.hyper_config["gamma"], self.hyper_config["use_igf"]
        
        D_item = np.array(item_counts + mu) ** -eps
        D_user = np.array(user_counts + mu) ** -eps
        D_item[np.isinf(D_item)] = 0.
        D_user[np.isinf(D_user)] = 0.

        norm_freq_matrix = X.multiply(D_item).multiply(D_user)
        norm_freq_matrix = torch.FloatTensor(norm_freq_matrix.toarray())

        # svd for normalized train matrix
        U, S, V = torch.svd_lowrank(norm_freq_matrix, q=k+200, niter=30)
        S = S / S.max() # max normalization

        # if use individualized graph filter
        if use_igf:
            # item set by user
            items_by_user = []
            for u in range(num_users):
                row_start = X.indptr[u]
                row_end = X.indptr[u + 1]
                items_by_user.append(X.indices[row_start:row_end])

            # user set by item
            users_by_item = []
            csc_matrix = X.tocsc()
            for i in range(num_items):
                col_start = csc_matrix.indptr[i]
                col_end = csc_matrix.indptr[i + 1]
                users_by_item.append(csc_matrix.indices[col_start:col_end])
            del csc_matrix

            # calculate homophily ratio
            homo_ratio_user, homo_ratio_item = [], []
            for u in range(num_users):
                user_counts = np.array(user_counts).flatten()
                if user_counts[u] > 1:
                    inter_items = torch.FloatTensor(X[:, items_by_user[u]].toarray()).T
                    inter_items[:, u] = 0
                    connect_matrix = torch.mm(inter_items, inter_items.T)
                    
                    size = inter_items.shape[0]
                    ratio_u = ((connect_matrix != 0).sum().item() - (connect_matrix.diag() != 0).sum().item()) / (size*(size-1))
                    homo_ratio_user.append(ratio_u)
                else:
                    homo_ratio_user.append(0)

            for i in range(num_items):
                item_counts = np.array(item_counts).flatten()
                if item_counts[i] > 1:
                    inter_users = torch.FloatTensor(X[users_by_item[i]].toarray())
                    inter_users[:, i] = 0
                    connect_matrix = torch.mm(inter_users, inter_users.T)

                    size = inter_users.shape[0]
                    ratio_i = ((connect_matrix != 0).sum().item() - (connect_matrix.diag() != 0).sum().item()) / (size*(size-1))
                    homo_ratio_item.append(ratio_i)
                else:
                    homo_ratio_item.append(0)

            homo_ratio_user = torch.Tensor(homo_ratio_user)
            homo_ratio_item = torch.Tensor(homo_ratio_item)

            # weight matrix
            weight_matrix = (U[:, :k] * self.individual_weight(S[:k], homo_ratio_user, beta, beta_end)) @ (V[:, :k] * self.individual_weight(S[:k], homo_ratio_item, beta, beta_end)).T
        else:
            # weight matrix
            weight_matrix = (U[:, :k] * S[:k].pow(beta)) @ (V[:, :k] * S[:k].pow(beta)).T
        weight_matrix = weight_matrix / (weight_matrix.sum(1)).unsqueeze(1)

        # norm gram matrix for all frequencies
        norm_freq_matrix = norm_freq_matrix @ norm_freq_matrix.T @ norm_freq_matrix
        norm_freq_matrix = norm_freq_matrix / (norm_freq_matrix.sum(1)).unsqueeze(1)

        # final matrix (m x n)
        self.prediction = (weight_matrix + gamma * norm_freq_matrix).sigmoid().numpy()

    def individual_weight(self, value, homo_ratio, beta, beta_end):
        y_min, y_max = beta, beta_end
        x_min, x_max = homo_ratio.min(), homo_ratio.max()
        homo_weight = (y_max - y_min) / (x_max - x_min) * homo_ratio + (x_max * y_min - y_max * x_min) / (x_max - x_min)
        homo_weight = homo_weight.unsqueeze(1)

        return value.pow(homo_weight)
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        user_ids = pck_users.cpu().numpy()

        full_preds = self.prediction[user_ids]
        full_preds = self._mask_predict(torch.from_numpy(full_preds), train_mask)

        return full_preds
