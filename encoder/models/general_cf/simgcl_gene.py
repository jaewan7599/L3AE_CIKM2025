import torch
import torch.nn as nn

from models.aug_utils import NodeMask
from models.general_cf.simgcl import SimGCL
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, ssl_con_loss
from models.text_embedding import load_text_embedding


class SimGCL_gene(SimGCL):
    def __init__(self, data_handler, configs):
        super(SimGCL_gene, self).__init__(data_handler, configs)

        # hyper-parameter
        self.mask_ratio, self.recon_weight, self.recon_tau = self.hyper_config['mask_ratio'], self.hyper_config['recon_weight'], self.hyper_config['recon_tau']

        # semantic-embedding
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

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # or nn.init.uniform

    def _mask(self):
        embeds = torch.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)

        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds
    
    def forward(self, adj=None, perturb=False, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj
        
        if not perturb:
            return super(SimGCL_gene, self).forward(adj, perturb=False)
        
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = torch.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = torch.concat([masked_user_embeds, masked_item_embeds], axis=0)
        embeds_list = [embeds]

        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list)

        return embeds[:self.user_num], embeds[self.user_num:]
    
    def _reconstruction(self, embeds, seeds):
        enc_embeds, llm_embeds  = self.mlp(embeds[seeds]), self.llm_embeds[seeds]
        recon_loss = ssl_con_loss(enc_embeds, llm_embeds, self.recon_tau)

        return recon_loss

    def cal_loss(self, batch_data):
        self.is_training = True

        masked_user_embeds, masked_item_embeds, seeds = self._mask()

        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True, masked_user_embeds=masked_user_embeds, masked_item_embeds=masked_item_embeds)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True, masked_user_embeds=masked_user_embeds, masked_item_embeds=masked_item_embeds)
        user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False, masked_user_embeds=masked_user_embeds, masked_item_embeds=masked_item_embeds)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]

        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_tau) + cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_tau)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        ### LLM loss
        recon_loss = self.recon_weight * self._reconstruction(torch.concat([user_embeds3, item_embeds3], axis=0), seeds)

        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + cl_loss + recon_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'recon_loss': recon_loss}

        return loss, losses
