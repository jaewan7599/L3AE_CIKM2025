import torch as t
from torch import nn


class BaseLinear(nn.Module):
    def __init__(self, configs):
        super(BaseLinear, self).__init__()

        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']

        self.hyper_config = configs['model']
    
    # suggest to return embeddings
    def forward(self):
        pass

    def cal_loss(self, batch_data):
        pass
    
    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask
    
    def full_predict(self, batch_data):
        """return all-rank predictions to evaluation process, should call _mask_predict for masking the training pairs

        Args:
            batch_data (tuple): data in a test batch, e.g. batch_users, train_mask
        
        Return:
            full_preds (torch.Tensor): a [test_batch_size * item_num] prediction tensor
        """
        pass