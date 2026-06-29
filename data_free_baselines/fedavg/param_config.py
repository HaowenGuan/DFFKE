import argparse
import logging
import os

import numpy as np
import torch
from data_free_baselines.datasets.data_distributor import DataDistributor
from lightfed.tools.funcs import consistent_hash, set_seed


def get_args(config_args=None, data_distributor=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--comm_round', type=int, default=100)

    parser.add_argument('--I', type=int, default=20, help='synchronization interval')

    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--eval_step_interval', type=int, default=5)

    parser.add_argument('--eval_batch_size', type=int, default=256)

    parser.add_argument('--lr_lm', type=float, default=0.008)

    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--model_type', type=str, default='Lenet',
                        choices=['Lenet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet18_32x32'])

    parser.add_argument('--model_norm', type=str, default='bn', choices=['none', 'bn', 'in', 'ln', 'gn'])

    parser.add_argument('--scale', type=lambda s: s == 'True', default=True)

    parser.add_argument('--mask', type=lambda s: s == 'True', default=False)

    parser.add_argument('--data_set', type=str, default='CIFAR10',
                        choices=['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'Tiny-Imagenet', 'FOOD101'])

    parser.add_argument('--data_partition_mode', type=str, default='non_iid_unbalanced',
                        choices=['iid', 'non_iid_unbalanced', 'non_iid_balanced'])

    parser.add_argument('--non_iid_alpha', type=float, default=0.01)

    parser.add_argument('--client_num', type=int, default=10)

    parser.add_argument('--selected_client_num', type=int, default=10)

    parser.add_argument('--device', type=torch.device, default='cuda')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=str, default='FedAvg')

    if config_args is None:
        args = parser.parse_args()
    else:
        # If config_args is given, parse does not read from command line
        args = parser.parse_args(args=[])
        # Load the configuration from a yaml file
        args.data_set = config_args['dataset']
        args.client_num = args.selected_client_num = config_args['n_clients']
        if len(config_args['models']) == 1:
            args.model_type = config_args['models'][0]
        else:
            raise "FedAvg does not support Heterogeneous Models"
        args.data_partition_mode = config_args['data_partition']
        args.non_iid_alpha = config_args['alpha']

    super_params = args.__dict__.copy()
    del super_params['log_level']
    super_params['device'] = super_params['device'].type
    ff = f"{args.app_name}-{consistent_hash(super_params, code_len=64)}.pkl"
    ff = f"{os.path.dirname(__file__)}/Result/{ff}"
    print(f"output file: {ff}")

    if data_distributor is None:
        args.data_distributor = _get_data_distributor(args)
    else:
        args.data_distributor = data_distributor

    return args


def _get_data_distributor(args):
    set_seed(args.seed + 5363)
    return DataDistributor(args)
