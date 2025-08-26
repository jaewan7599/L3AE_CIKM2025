from config.configurator import configs
from trainer.trainer import init_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner
from utils.utils import save_neural_result

import warnings, time
warnings.filterwarnings("ignore")


def main():
    init_seed(configs)
    data_handler = build_data_handler(configs)
    data_handler.load_data()

    model = build_model(data_handler, configs).to(configs['device'])
    
    logger = Logger(configs)
    log_file_path = logger.file_path

    trainer = build_trainer(data_handler, configs, logger)

    train_start = time.time()
    valid_result, test_result = trainer.train(model, False)
    train_end = time.time()

    print("============ Valid Result ============")
    print(valid_result)

    print("============ Test Result ============")
    print(test_result)

    save_neural_result(False, configs, int(train_end - train_start), log_file_path, valid_result, test_result)

main()


