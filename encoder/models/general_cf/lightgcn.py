import torch
from torch import nn
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
# from models.model_utils import SpAdjEdgeDrop


class LightGCN(BaseModel):
    def __init__(self, data_handler, configs):
        super(LightGCN, self).__init__(configs)

        self.adj = data_handler.torch_adj

        self.keep_rate = configs['model']['keep_rate']
        self.final_embeds, self.is_training = None, False

        self.user_embeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.item_num, self.embedding_size)))

        # hyper-parameter
        self.layer_num, self.reg_weight = self.hyper_config['layer_num'], self.hyper_config['reg_weight']
    
    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)
    
    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj

        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        
        embeds = torch.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]

        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        self.final_embeds = sum(embeds_list) / len(embeds_list)

        return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
    
    def cal_loss(self, batch_data):
        self.is_training = True
        
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds, pos_embeds, neg_embeds = user_embeds[ancs], item_embeds[poss], item_embeds[negs]
        
        loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

        reg_loss = self.reg_weight * reg_params(self)
        loss = loss + reg_loss
        losses = {'rec_loss': loss, 'reg_loss': reg_loss}

        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False

        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        pck_user_embeds = user_embeds[pck_users]

        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        
        return full_preds
    
    def ego_embeddings(self):
        user_embed, item_embed = self.user_embeds, self.item_embeds

        return user_embed, item_embed
    