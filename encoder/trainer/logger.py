import os, logging, datetime


def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')


class Logger(object):
    def __init__(self, configs, log_configs=True):
        self.configs = configs

        model_name = self.configs['model']['name']
        log_dir_path = './encoder/log/{}'.format(model_name)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

        self.logger, dataset_name, self.local_time = logging.getLogger('train_logger'), configs['data']['name'], get_local_time()
        self.logger.setLevel(logging.INFO)

        if not configs['tune']['enable']:
            file_path = f'{log_dir_path}/{dataset_name}_{self.local_time}.log'
            log_file = logging.FileHandler(file_path, 'a', encoding='utf-8')
        else:
            file_path = f'{log_dir_path}/{dataset_name}-tune_{self.local_time}.log'
            log_file = logging.FileHandler(file_path, 'a', encoding='utf-8')

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)
        self.file_path = file_path

        if log_configs:
            tmp_configs = {}
            tmp_configs['optimizer'] = configs['optimizer']
            tmp_configs['train'] = configs['train']
            tmp_configs['test'] = configs['test']
            tmp_configs['data'] = configs['data']
            tmp_configs['model'] = configs['model']

            self.log(tmp_configs)

    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)

        if print_to_console:
            print(message)

    def log_loss(self, epoch_idx, loss_log_dict, save_to_log=True, print_to_console=True):
        epoch = self.configs['train']['epoch']
        message = '[Epoch {:3d} / {:3d}] '.format(epoch_idx, epoch)
        for loss_name in loss_log_dict:
            message += '{}: {:.4f} '.format(loss_name, loss_log_dict[loss_name])

        if save_to_log:
            self.logger.info(message)

        if print_to_console:
            print(message)

    def log_eval(self, eval_result, k, data_type, save_to_log=True, print_to_console=True, epoch_idx=None):
        if epoch_idx is not None:
            message = 'Epoch {:3d} {} '.format(epoch_idx, data_type)
        else:
            message = '{} '.format(data_type)

        for metric in eval_result:
            message += '['
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '

        if save_to_log:
            self.logger.info(message)

        if print_to_console:
            print(message)
