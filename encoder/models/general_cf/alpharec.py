import torch
import torch.nn as nn
import torch.nn.functional as F

from models.general_cf.lightgcn import LightGCN
from models.loss_utils import reg_params, cal_infonce_loss
from models.text_embedding import load_text_embedding


class AlphaRec(LightGCN):
    def __init__(self, data_handler, configs):
        super(AlphaRec, self).__init__(data_handler, configs)

        # hyper-parameter
        self.cl_tau, self.no_pred_norm = configs['model']['cl_tau'], configs['model']['no_pred_norm']

        # semantic-embeddings
        dataset, llm = configs['data']['name'], configs['data']['llm']
        user_llm_embeds, item_llm_embeds = load_text_embedding(dataset, llm, data_handler.trn_mat)
        self.user_llm_embeds, self.item_llm_embeds = torch.tensor(user_llm_embeds).float().cuda(), torch.tensor(item_llm_embeds).float().cuda()
        self.llm_embed_size = self.user_llm_embeds.shape[1]

        self.build_adapter()
        self.adapter.apply(self._init_weights)
    
    def build_adapter(self):
        self.adapter = nn.Sequential(
            nn.Linear(self.llm_embed_size, int(self.llm_embed_size / 2)),
            nn.LeakyReLU(),
            nn.Linear(int (self.llm_embed_size / 2), self.embedding_size))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj

        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        user_embed, item_embed = self.adapter(self.user_llm_embeds), self.adapter(self.item_llm_embeds)

        embeds = torch.concat([user_embed, item_embed], axis=0)
        embeds_list = [embeds]

        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        self.final_embeds = sum(embeds_list) / len(embeds_list)

        return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

    def cal_loss(self, batch_data):
        self.is_training = True

        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        users, poss, negs = batch_data
        users_emb, poss_emb, _ = user_embeds[users], item_embeds[poss], item_embeds[negs]

        ssm_loss = cal_infonce_loss(users_emb, poss_emb, poss_emb, self.cl_tau)
        ssm_loss /= users_emb.shape[0]
        
        reg_loss = self.reg_weight * reg_params(self)

        loss = ssm_loss + reg_loss
        losses = {'ssm_loss': ssm_loss}

        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False

        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        if not self.no_pred_norm:
            user_embeds = F.normalize(user_embeds, dim=-1)
            item_embeds = F.normalize(item_embeds, dim=-1)

        pck_user_embeds = user_embeds[pck_users]

        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds

    def ego_embeddings(self):
        user_embed, item_embed = self.adapter(self.user_llm_embeds), self.adapter(self.item_llm_embeds)

        return user_embed, item_embed