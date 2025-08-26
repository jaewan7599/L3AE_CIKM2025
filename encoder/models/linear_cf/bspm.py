from models.base_linear import BaseLinear
import torch
import numpy as np
from scipy import sparse
from torchdiffeq import odeint
import scipy.sparse.linalg as linalg


class BSPM(BaseLinear):
    def __init__(self, data_handler, configs):
        super(BSPM, self).__init__(configs)
        
        self.trn_mat = data_handler.trn_mat.tocsr()
        
        self.merge = self.hyper_config['merge'] # EM: final_sharping=True, LM: final_sharping=False
        self.sharpen_solver = self.hyper_config['solver_shr'] # rk4, euler
        sharpen_T, sharpen_K = self.hyper_config['T_s'], self.hyper_config['K_s']

        self.ode_times = torch.linspace(0, 1, 2)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1)

        self.t_point_combination = self.hyper_config['t_point_combination']
        self.beta = self.hyper_config['beta']
        self.factor_dim = self.hyper_config['factor_dim'] # {256, 384, 448}
        
        self.__init_weight()
        
    def __init_weight(self):
        adj_mat = self.trn_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sparse.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sparse.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sparse.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        
        ut, s, self.vt = linalg.svds(self.norm_adj, self.factor_dim)

    def IDLFunction(self, t, r):
        out = r.numpy() @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
        out = out - r.numpy()
        return torch.Tensor(out)

    def blurFunction(self, t, r):
        R = self.norm_adj
        out = r.numpy() @ R.T @ R
        out = out - r.numpy()
        return torch.Tensor(out)
    
    def sharpenFunction(self, t, r):
        R = self.norm_adj
        out = r.numpy() @ R.T @ R
        return torch.Tensor(-out)

    def inference(self, batch_matrix):
        with torch.no_grad():
            blurred_out = odeint(func=self.blurFunction, y0=torch.Tensor(batch_matrix), t=self.ode_times, method='euler')
            idl_out = odeint(func=self.IDLFunction, y0=torch.Tensor(batch_matrix), t=self.ode_times, method='euler')
            
            if self.merge == 'EM':
                sharpened_out = odeint(func=self.sharpenFunction, y0=self.beta * idl_out[-1] + blurred_out[-1], t=self.sharpening_times, method=self.sharpen_solver)
            else:
                sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out[-1], t=self.sharpening_times, method=self.sharpen_solver)
        
        if self.t_point_combination:
            U_2 = torch.mean(torch.cat([blurred_out[1:,...], sharpened_out[1:,...]], axis=0), axis=0)
        else:
            U_2 = sharpened_out[-1]
        
        if self.merge == 'EM':
            ret = U_2.numpy()
        else:
            ret = U_2.numpy() + self.beta * idl_out[-1].numpy()
        
        return ret

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        user_ids = pck_users.cpu().numpy()
        
        input_matrix = np.array(self.trn_mat[user_ids].toarray())
        eval_output = self.inference(input_matrix)
        
        full_preds = torch.FloatTensor(eval_output)
        full_preds = self._mask_predict(full_preds, train_mask)
        
        return full_preds

