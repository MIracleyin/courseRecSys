import argparse
import os
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation, save_split_dataloaders
from recbole.utils import init_logger, get_trainer, init_seed, set_color, get_model
from recbole.data import create_dataset

from model import model_name_map
from data.dataset import GeneralGraphDataset, SequentialGraphDataset

def run(model=None, dataset='mooc', config_file_list=None, saved=True):
    current_path = os.path.dirname(os.path.realpath(__file__))
    # base config file
    overall_init_file = os.path.join(current_path, 'config/overall.yaml')
    # model config file
    model_init_file = os.path.join(current_path, 'config/model/' + model + '.yaml')
    # dataset config file
    dataset_init_file = os.path.join(current_path, 'config/dataset/' + dataset + '.yaml')

    config_file_list = [overall_init_file, model_init_file, dataset_init_file]  # env model dataset

    model_class = model_name_map.get(model) or model
    config = Config(model=model_class, dataset=dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    # dataset = create_dataset(config) # only for GRU4Rec
    dataset = SequentialGraphDataset(config) # for TP-GNN
    # dataset = GeneralGraphDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, action='store', help="model name")
    args, _ = parser.parse_known_args()

    model_name = args.model
    run(model_name)