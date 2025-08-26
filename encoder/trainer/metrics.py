import torch
import numpy as np
from tqdm import tqdm


class Metric(object):
    def __init__(self, configs):
        self.configs = configs
        self.metrics = self.configs['test']['metrics']
        self.k = self.configs['test']['k']
    
    def recall(self, test_data, r, k, pscore=None, inv_pscore_sum=None):
        """
            test_data: list of ground truth items
            r: batch_size x max(top_k)
            k: top_k
            pscore: item propensity score (batch_size x max(top_k))
            inv_pscore_sum: inverse pscore sum (batch_size, )
        """
        if pscore is None:
            right_pred = r[:, :k].sum(1)
            recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
        else: # unbaised recall
            pred_data = r[:, :k]
            pscore = pscore[:, :k]
            norm_pred_data = pred_data / pscore

            right_pred = np.sum(norm_pred_data, axis=1)
            recall_n = inv_pscore_sum

        recall = np.sum(right_pred / recall_n)

        return recall

    def precision(self, r, k):
        right_pred = r[:, :k].sum(1)
        precis_n = k
        precision = np.sum(right_pred) / precis_n

        return precision

    def mrr(self, r, k):
        pred_data = r[:, :k]
        scores = 1. / np.arange(1, k + 1)
        pred_data = pred_data * scores
        pred_data = pred_data.sum(1)

        return np.sum(pred_data)

    def ndcg(self, test_data, r, k, pscore=None, inv_pscore_sum=None):
        assert len(r) == len(test_data)
        if pscore is None:
            pred_data = r[:, :k]
            test_matrix = np.zeros((len(pred_data), k))
            for i, items in enumerate(test_data):
                length = k if k <= len(items) else len(items)
                test_matrix[i, :length] = 1
            max_r = test_matrix
            idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
            dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
            dcg = np.sum(dcg, axis=1)
            idcg[idcg == 0.] = 1.
            ndcg = dcg / idcg
            ndcg[np.isnan(ndcg)] = 0.
        else: # unbiased ndcg
            pred_data = r[:, :k]
            test_matrix = np.zeros((len(pred_data), k))
            for i, items in enumerate(test_data):
                length = k if k <= len(items) else len(items)
                test_matrix[i, :length] = 1
            max_r = test_matrix
            idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
            
            pscore = pscore[:, :k]
            norm_dcg = (1. / np.log2(np.arange(2, k + 2))) / pscore
            masked_dcg = pred_data * norm_dcg
            dcg = np.sum(masked_dcg, axis=1)
            idcg[idcg == 0.] = 1.
            ndcg = dcg / idcg
            ndcg[np.isnan(ndcg)] = 0.

            # normalization for unbiased metrics
            ndcg /= inv_pscore_sum

        return np.sum(ndcg)
    
    def get_label(self, test_data, pred_data):
        r = []
        for i in range(len(test_data)):
            ground_true = test_data[i]
            predict_topk = pred_data[i]
            pred = list(map(lambda x: x in ground_true, predict_topk))
            pred = np.array(pred).astype("float")
            r.append(pred)
        return np.array(r).astype('float')

    def eval_batch(self, data, topks):
        sorted_items = data[0].numpy()
        ground_true = data[1]
        r = self.get_label(ground_true, sorted_items)

        if len(data) == 4:
            pscore = data[2]
            inv_pscore_sum = data[3]
        else:
            pscore = None
            inv_pscore_sum = None

        result = {}
        for metric in self.metrics:
            result[metric] = []

        for k in topks:
            for metric in result:
                if metric == 'recall':
                    result[metric].append(self.recall(ground_true, r, k, pscore, inv_pscore_sum))
                if metric == 'ndcg':
                    result[metric].append(self.ndcg(ground_true, r, k, pscore, inv_pscore_sum))
                if metric == 'precision':
                    result[metric].append(self.precision(r, k))
                if metric == 'mrr':
                    result[metric].append(self.mrr(r, k))

        for metric in result:
            result[metric] = np.array(result[metric])

        return result
    
    def filtered_eval_batch(self, data, topks):
        sorted_items = data[0].numpy()
        ground_true = data[1]
        
        valid_indices = []
        valid_ground_true = []
        valid_sorted_items = []
        
        for i, gt in enumerate(ground_true):
            if gt:
                valid_indices.append(i)
                valid_ground_true.append(gt)
                valid_sorted_items.append(sorted_items[i])
        
        if not valid_indices:
            result = {}
            for metric in self.metrics:
                result[metric] = np.zeros(len(topks))
            return result
        
        valid_sorted_items = np.array(valid_sorted_items)
        r = self.get_label(valid_ground_true, valid_sorted_items)
        
        result = {}
        for metric in self.metrics:
            result[metric] = []
        
        for k in topks:
            for metric in result:
                if metric == 'recall':
                    result[metric].append(self.recall(valid_ground_true, r, k))
                if metric == 'ndcg':
                    result[metric].append(self.ndcg(valid_ground_true, r, k))
                if metric == 'precision':
                    result[metric].append(self.precision(r, k))
                if metric == 'mrr':
                    result[metric].append(self.mrr(r, k))
        
        for metric in result:
            result[metric] = np.array(result[metric])
        
        return result

    def eval(self, model, test_dataloader, item_filter=None):
        result = {}
        for metric in self.metrics:
            result[metric] = np.zeros(len(self.k))

        batch_ratings, ground_truths, test_user_count = [], [], 0
        test_user_num = len(test_dataloader.dataset.test_users)

        for _, tem in enumerate(tqdm(test_dataloader, desc='Testing', ncols=100)):
            if not isinstance(tem, list):
                tem = [tem]

            test_user = tem[0].numpy().tolist()
            batch_data = list(map(lambda x: x.long().to(self.configs['device']), tem))

            # predict result
            with torch.no_grad():
                batch_pred = model.full_predict(batch_data)
            test_user_count += batch_pred.shape[0]

            # filter out history items
            batch_pred = self._mask_history_pos(batch_pred, test_user, test_dataloader)
            _, batch_rate = torch.topk(batch_pred, k=max(self.k))

            batch_ratings.append(batch_rate.cpu())

            # ground truth
            ground_truth = []
            for user_idx in test_user:
                ground_truth.append(list(test_dataloader.dataset.user_pos_lists[user_idx]))
            ground_truths.append(ground_truth)
        assert test_user_count == test_user_num

        # calculate metrics
        data_pair = zip(batch_ratings, ground_truths)
        eval_results = []
        for _data in data_pair:
            eval_results.append(self.eval_batch(_data, self.k))
            
        for batch_result in eval_results:
            for metric in self.metrics:
                result[metric] += batch_result[metric] / test_user_num

        return result

    def filtered_eval(self, model, test_dataloader, item_filter):
        result = {}
        for metric in self.metrics:
            result[metric] = np.zeros(len(self.k))

        batch_ratings = []
        ground_truths = []
        test_user_count = 0
        item_filter_set = set(item_filter)

        for _, tem in enumerate(tqdm(test_dataloader, desc="Testing", ncols=100)):
            if not isinstance(tem, list):
                tem = [tem]
            test_user = tem[0].numpy().tolist()
            batch_data = list(map(lambda x: x.long().to(self.configs['device']), tem))
            
            with torch.no_grad():
                batch_pred = model.full_predict(batch_data)
            test_user_count += batch_pred.shape[0]
            
            batch_pred = self._mask_history_pos(batch_pred, test_user, test_dataloader)
            
            _, batch_rate = torch.topk(batch_pred, k=max(self.k))
            batch_ratings.append(batch_rate.cpu())
            
            ground_truth = []
            for user_idx in test_user:
                gt_items = list(test_dataloader.dataset.user_pos_lists[user_idx])
                gt_items = [item for item in gt_items if item in item_filter_set]
                ground_truth.append(gt_items)
            ground_truths.append(ground_truth)
        
        valid_user_count = 0
        for batch_gt in ground_truths:
            for user_gt in batch_gt:
                if user_gt:
                    valid_user_count += 1

        # calculate metrics
        data_pair = zip(batch_ratings, ground_truths)
        eval_results = []
        for _data in data_pair:
            eval_results.append(self.filtered_eval_batch(_data, self.k))
        
        for batch_result in eval_results:
            for metric in self.metrics:
                result[metric] += batch_result[metric] / valid_user_count if valid_user_count > 0 else 0

        return result
    
    def unbiased_eval(self, model, test_dataloader, pscore):
        result = {}
        for metric in self.metrics:
            result[metric] = np.zeros(len(self.k))

        batch_pscores = []
        batch_inv_pscore_sums = []
        batch_ratings = []
        ground_truths = []
        test_user_count = 0
        test_user_num = len(test_dataloader.dataset.test_users)

        for _, tem in enumerate(test_dataloader):
            if not isinstance(tem, list):
                tem = [tem]
            test_user = tem[0].numpy().tolist()
            batch_data = list(map(lambda x: x.long().to(self.configs['device']), tem))
            # predict result
            with torch.no_grad():
                batch_pred = model.full_predict(batch_data)
            test_user_count += batch_pred.shape[0]
            # filter out history items
            batch_pred = self._mask_history_pos(batch_pred, test_user, test_dataloader)
            _, batch_rate = torch.topk(batch_pred, k=max(self.k))
            batch_pscore = pscore[batch_rate.cpu()]
            batch_pscores.append(batch_pscore)
            batch_ratings.append(batch_rate.cpu())
            # ground truth
            ground_truth = []
            for user_idx in test_user:
                ground_truth.append(list(test_dataloader.dataset.user_pos_lists[user_idx]))
            ground_truths.append(ground_truth)
            # inverse_pscore_sum
            batch_inv_pscore_sum = []
            for user_gt in ground_truth:
                user_pscores = pscore[user_gt]
                user_inv_pscores_sum = (1 / user_pscores).sum()
                batch_inv_pscore_sum.append(user_inv_pscores_sum)
            batch_inv_pscore_sum = np.array(batch_inv_pscore_sum, dtype=np.float64)
            batch_inv_pscore_sums.append(batch_inv_pscore_sum)

        assert test_user_count == test_user_num

        # calculate metrics
        data_pair = zip(batch_ratings, ground_truths, batch_pscores, batch_inv_pscore_sums)
        eval_results = []
        for _data in data_pair:
            eval_results.append(self.eval_batch(_data, self.k))
        for batch_result in eval_results:
            for metric in self.metrics:
                result[metric] += batch_result[metric] / test_user_num

        return result

    def eval_save(self, model, test_dataloader):
        result = {}
        for metric in self.metrics:
            result[metric] = np.zeros(len(self.k))

        batch_ratings = []
        ground_truths = []
        test_user_count = 0
        test_user_num = len(test_dataloader.dataset.test_users)
        candidate_set = {}
        for _, tem in enumerate(test_dataloader):
            if not isinstance(tem, list):
                tem = [tem]
            test_user = tem[0].numpy().tolist()
            batch_data = list(map(lambda x: x.long().to(self.configs['device']), tem))
            # predict result
            with torch.no_grad():
                batch_pred = model.full_predict(batch_data)
            test_user_count += batch_pred.shape[0]
            # filter out history items
            batch_pred = self._mask_history_pos(batch_pred, test_user, test_dataloader)
            _, batch_rate = torch.topk(batch_pred, k=100)
            batch_ratings.append(batch_rate.cpu())
            # ground truth
            ground_truth = []
            for user_idx in test_user:
                ground_truth.append(list(test_dataloader.dataset.user_pos_lists[user_idx]))
            for i in range(len(test_user)):
                user_idx = test_user[i]
                candidate_set[user_idx] = batch_rate[i].detach().cpu().numpy().tolist()
            ground_truths.append(ground_truth)
        assert test_user_count == test_user_num
        assert len(candidate_set) == test_user_num

        # calculate metrics
        data_pair = zip(batch_ratings, ground_truths)
        eval_results = []
        for _data in data_pair:
            eval_results.append(self.eval_batch(_data, self.k))
        for batch_result in eval_results:
            for metric in self.metrics:
                result[metric] += batch_result[metric] / test_user_num

        return result, candidate_set

    def _mask_history_pos(self, batch_rate, test_user, test_dataloader):
        if not hasattr(test_dataloader.dataset, 'user_history_lists'):
            return batch_rate
        for i, user_idx in enumerate(test_user):
            pos_list = test_dataloader.dataset.user_history_lists[user_idx]
            batch_rate[i, pos_list] = -np.inf
        return batch_rate

    def filter_eval_result(self, eval_result, item_filter):
        filtered_result = {}
        for metric in self.metrics:
            filtered_result[metric] = np.zeros(len(self.k))
        
        return filtered_result