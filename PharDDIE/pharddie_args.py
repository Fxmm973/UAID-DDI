import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config_loader import load_config


def read_options():
    cfg = load_config('pharddie.json')
    parser = argparse.ArgumentParser(description='PharDDIE few-shot training/evaluation')
    parser.add_argument("--dataset", default=cfg.get('dataset', 'dataset1'), type=str)
    parser.add_argument("--embed_dim", default=cfg.get('embed_dim', 128), type=int)
    parser.add_argument("--few", default=cfg.get('few', 10), type=int)
    parser.add_argument("--batch_size", default=cfg.get('batch_size', 256), type=int)
    parser.add_argument("--neg_num", default=cfg.get('neg_num', 1), type=int)
    parser.add_argument("--random_embed", default=cfg.get('random_embed', False), type=bool)
    parser.add_argument("--train_few", default=cfg.get('train_few', 10), type=int)
    parser.add_argument("--lr", default=cfg.get('lr', 0.001), type=float)
    parser.add_argument("--max_batches", default=cfg.get('max_batches', 40000), type=int)
    parser.add_argument("--dropout", default=cfg.get('dropout', 0.2), type=float)
    parser.add_argument("--log_every", default=cfg.get('log_every', 50), type=int)
    parser.add_argument("--eval_every", default=cfg.get('eval_every', 1000), type=int)
    parser.add_argument("--fine_tune", default=cfg.get('fine_tune', True), type=bool)
    parser.add_argument("--aggregate", default=cfg.get('aggregate', 'max'), type=str)
    parser.add_argument("--max_neighbor", default=cfg.get('max_neighbor', 30), type=int)
    parser.add_argument("--no_meta", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--grad_clip", default=cfg.get('grad_clip', 5.0), type=float)
    parser.add_argument("--weight_decay", default=cfg.get('weight_decay', 0.0), type=float)
    parser.add_argument("--embed_model", default=cfg.get('embed_model', 'TransE'), type=str)
    parser.add_argument("--prefix", default=cfg.get('prefix', 'intial'), type=str)
    parser.add_argument("--seed", default=cfg.get('seed', 19940419), type=int)

    args = parser.parse_args()
    args.save_path = 'models/' + args.prefix

    print("------HYPERPARAMETERS-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print("----------------------------")

    return args
