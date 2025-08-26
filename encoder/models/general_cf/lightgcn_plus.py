import torch
from torch import nn

from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.general_cf.lightgcn import LightGCN
from models.text_embedding import load_text_embedding
# from models.model_utils import SpAdjEdgeDrop


class LightGCN_plus(LightGCN):
    def __init__(self, data_handler, configs):
        super(LightGCN_plus, self).__init__(data_handler, configs)

        # hyper-parameter
        self.kd_weight, self.kd_tau = self.hyper_config['kd_weight'], self.hyper_config['kd_tau']

        # semantic-embeddings
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
                nn.init.xavier_uniform_(m.weight)  # or nn.init.uniform
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds, pos_embeds, neg_embeds = user_embeds[ancs], item_embeds[poss], item_embeds[negs]

        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True

        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

        user_llm_embeds, item_llm_embeds = self.mlp(self.user_llm_embeds), self.mlp(self.item_llm_embeds)
        anc_llm_embeds, pos_llm_embeds, neg_llm_embeds = self._pick_embeds(user_llm_embeds, item_llm_embeds, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        
        kd_loss = cal_infonce_loss(anc_embeds, anc_llm_embeds, user_llm_embeds, self.kd_tau) + \
                  cal_infonce_loss(pos_embeds, pos_llm_embeds, pos_llm_embeds, self.kd_tau) + \
                  cal_infonce_loss(neg_embeds, neg_llm_embeds, neg_llm_embeds, self.kd_tau)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        loss = bpr_loss + reg_loss + kd_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'kd_loss': kd_loss}

        return loss, losses
