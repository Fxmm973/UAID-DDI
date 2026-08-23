import sys, os
os.environ['TQDM_DISABLE'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random, logging
import numpy as np
import torch

sys.argv = ['eviddie_trainer.py', '--dataset', 'dataset1', '--few', '10', '--train_few', '10',
            '--batch_size', '256', '--max_batches', '2', '--seed', '19940419', '--prefix', 'smoke_kg']

from eviddie_args import read_options
from eviddie_trainer import Trainer, EvidentialLoss
import eviddie_trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: - %(message)s')

args = read_options()
args.save_path = f'models/{args.prefix}_seed{args.seed}'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

trainer = Trainer(args)
eviddie_trainer.loss_fn = EvidentialLoss(annealing_step=10000)
trainer.train()
print('SMOKE OK: __init__ + build_connection + 2 training batches completed without crash')
