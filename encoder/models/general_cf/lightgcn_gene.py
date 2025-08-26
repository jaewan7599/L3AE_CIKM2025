import torch
from torch import nn

from models.aug_utils import NodeMask
from models.general_cf.lightgcn import LightGCN
# from models.model_utils import SpAdjEdgeDrop
from models.loss_utils import cal_bpr_loss, reg_params, ssl_con_loss
from models.text_embedding import load_text_embedding


class LightGCN_gene(LightGCN):
    def __init__(self, data_handler, configs):
        super(LightGCN_gene, self).__init__(data_handler, configs)

        # hyper-parameter
        self.mask_ratio, self.recon_weight, self.recon_tau = self.hyper_config['mask_ratio'], self.hyper_config['recon_weight'], self.hyper_config['recon_tau']

        # semantic-embeddings
        dataset, llm = configs['data']['name'], configs['data']['llm']
        user_llm_embeds, item_llm_embeds = load_text_embedding(dataset, llm, data_handler.trn_mat)
        user_llm_embeds, item_llm_embeds = torch.tensor(user_llm_embeds).float().cuda(), torch.tensor(item_llm_embeds).float().cuda()
        self.llm_embeds = torch.concat([user_llm_embeds, item_llm_embeds], dim=0)
        _, llm_dim = user_llm_embeds.shape

        # generative process
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, (llm_dim + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((llm_dim + self.embedding_size) // 2, llm_dim))

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)
    
    def _mask(self):
        embeds = torch.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)

        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds

    def forward(self, adj=None, keep_rate=1.0, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj

        if not self.is_training and self.final_embeds is not None:
            return super(LightGCN_gene, self).forward(adj, 1.0)
        
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = torch.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = torch.concat([masked_user_embeds, masked_item_embeds], axis=0)
        embeds_list = [embeds]
        
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)

        return embeds[:self.user_num], embeds[self.user_num:]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds, pos_embeds, neg_embeds = user_embeds[ancs], item_embeds[poss], item_embeds[negs]

        return anc_embeds, pos_embeds, neg_embeds

    def _reconstruction(self, embeds, seeds):
        enc_embeds, llm_embeds  = self.mlp(embeds[seeds]), self.llm_embeds[seeds]
        recon_loss = ssl_con_loss(enc_embeds, llm_embeds, self.recon_tau)

        return recon_loss

    def cal_loss(self, batch_data):
        self.is_training = True

        masked_user_embeds, masked_item_embeds, seeds = self._mask()

        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate, masked_user_embeds, masked_item_embeds)
        ancs, poss, negs = batch_data
        anc_embeds, pos_embeds, neg_embeds = user_embeds[ancs], item_embeds[poss], item_embeds[negs]

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

        reg_loss = self.reg_weight * reg_params(self)

        recon_loss = self.recon_weight * self._reconstruction(torch.concat([user_embeds, item_embeds], axis=0), seeds)

        loss = bpr_loss + reg_loss + recon_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'recon_loss': recon_loss}

        return loss, losses
