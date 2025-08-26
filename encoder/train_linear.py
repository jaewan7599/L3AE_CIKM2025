from config.linear_configurator import configs
from trainer.trainer import init_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner
import time, warnings
from utils.utils import save_linear_result
warnings.filterwarnings("ignore")


def main():
    init_seed(configs)
    data_handler = build_data_handler(configs)
    data_handler.load_data()
    print(configs)

    train_start = time.time()
    model = build_model(data_handler, configs).to(configs['device'])
    train_end = time.time()

    trainer = build_trainer(data_handler, configs)

    valid_result, test_result = trainer.train(model, False)

    print("============ Valid Result ============")
    print(valid_result)

    print("============ Test Result ============")
    print(test_result)

    save_linear_result(False, configs, int(train_end - train_start), valid_result, test_result)


main()


