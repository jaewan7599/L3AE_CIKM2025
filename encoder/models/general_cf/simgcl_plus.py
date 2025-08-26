import torch
from torch import nn

from models.general_cf.simgcl import SimGCL
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.text_embedding import load_text_embedding


class SimGCL_plus(SimGCL):
    def __init__(self, data_handler, configs):
        super(SimGCL_plus, self).__init__(data_handler, configs)

        # hyper-parameter
        self.kd_weight, self.kd_tau = self.hyper_config['kd_weight'], self.hyper_config['kd_tau']

        # semantic-embedding
        dataset, llm = configs['data']['name'], configs['data']['llm']
        user_llm_embeds, item_llm_embeds = load_text_embedding(dataset, llm, data_handler.trn_mat)
        self.user_llm_embeds, self.item_llm_embeds = torch.tensor(user_llm_embeds).float().cuda(), torch.tensor(item_llm_embeds).float().cuda()
        _, llm_dim = self.user_llm_embeds.shape

        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, (llm_dim + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((llm_dim + self.embedding_size) // 2, self.embedding_size))

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # or nn.init.uniform

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)
        user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False)

        ### CF loss
        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]

        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_tau) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_tau)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        ### LLM loss
        user_llm_embeds, item_llm_embeds = self.mlp(self.user_llm_embeds), self.mlp(self.item_llm_embeds)
        anc_llm_embeds, pos_llm_embeds, neg_llm_embeds = self._pick_embeds(user_llm_embeds, item_llm_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds3, anc_llm_embeds, user_llm_embeds, self.kd_tau) + \
                  cal_infonce_loss(pos_embeds3, pos_llm_embeds, pos_llm_embeds, self.kd_tau) + \
                  cal_infonce_loss(neg_embeds3, neg_llm_embeds, neg_llm_embeds, self.kd_tau)
        kd_loss /= anc_embeds3.shape[0]
        kd_loss *= self.kd_weight

        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + cl_loss + kd_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'kd_loss': kd_loss}

        return loss, losses
