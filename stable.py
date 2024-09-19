from Lero.create_training_file import create_training_file
from Lero.lero_train import training_pairwise
from Lero.lero_test import test
import argparse, os, json
import logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='stats')
parser.add_argument('--logger_file', type=str, default='results/log.txt')
args = parser.parse_args()
logger.setup_logger(args.logger_file)

if __name__ == '__main__':
    latency_dir = f'results/{args.dataset}'
    # os.makedirs(latency_dir, exist_ok=True)
    # train_path = os.path.join('data/labeled_train_data', f'{args.dataset}_labeled.txt')
    # train_file = f'data/lero_train_files/{args.dataset}_labeled_train_file'
    save_path = f"saved_models/{args.dataset}/stable_model"
    test_file = f"data/test/{args.dataset}_test.txt"

    # os.makedirs(os.path.dirname(train_file), exist_ok=True)
    # create_training_file(train_file, train_path)
    # training_pairwise(None, save_path, train_file)
    test(save_path, test_file, latency_dir)
    with open(os.path.join(latency_dir, 'lero_dict.json'), 'r') as f:
        latency = json.load(f)['sum'] / 1000
        print(f'test latency:{latency}')
        
    with open(os.path.join(latency_dir, 'lero_latency'), 'w') as f:
        json.dump(latency, f, indent=4)


