import os
import yaml
import wandb
import random
import argparse
import numpy as np
import torch

from DFFKE import data_free_federated_knowledge_exchange
from dataset.utils_dataset import DataDistributor
from baseline_main import run_baseline


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_experiment', type=str, default=None, help='group experiment name (or folder name)')
    parser.add_argument('--config_file', type=str, default='DFFKE.yaml', help='path to the config file')
    run_args = parser.parse_args()

    current_folder = os.path.dirname(os.path.abspath(__file__))
    if run_args.group_experiment:
        main_config_file = f'{current_folder}/configs/{run_args.group_experiment}/__init__.yaml'
        # Load the main config file, the general settings of the group experiment
        with open(main_config_file, 'r') as file:
            args = yaml.safe_load(file)
        config_file = f'{current_folder}/configs/{run_args.group_experiment}/{run_args.config_file}'
        # Load the specific config file for the algorithm
        with open(config_file, 'r') as file:
            sub_args = yaml.safe_load(file)
        for key, value in sub_args.items():
            args[key] = sub_args[key]
    else:
        # Load the specific config file for the experiment
        config_file = current_folder + f'/configs/{run_args.config_file}'
        with open(config_file, 'r') as file:
            args = yaml.safe_load(file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device_id'])
    if not torch.cuda.is_available():
        args['device'] = 'cpu'
        print('CUDA is not available. Using CPU.')
    else:
        print(f'Using CUDA:{args["device_id"]}')

    return args


def generate_wandb_name(args):
    name = f'{args["algorithm"]} / '
    if args["algorithm"] == 'DFFKE' and args['knowledge_exchange_rounds'] > 0:
        name += args['generator_model'] + ' / '
        name += 'DFFKE'
        if args['local_align_after_knowledge_exchange']:
            name += ' + Review'
        if args['new_client_opt_every_round']:
            name += ' + NewOpt'
        name += ' / '
        name += f'L2G {args["L2G_epoch"]}.emb.kl.lmd / '
        name += f'G2L {args["G2L_epoch"]}'
        if args['G2L_local_review_iteration']:
            name += f'.{args["G2L_local_review_iteration"] if args["G2L_local_review_iteration"] > 1 else ""}'
            if args['G2L_augment_real']:
                name += 'aug'
            name += f'real.'
        if args['G2L_cal_emb_loss']:
            name += 'emb.'
        name += 'kl / '
        if args['dock_kd']:
            name += 'Dock KD / '
        else:
            name += 'EMB KD / '
    name += f'{args["n_clients"]} Clients / '
    name += f'{args["dataset"]} Dir({args["alpha"]})'
    return name


def init_wandb(args):
    wandb.init(
        sync_tensorboard=False,
        project=args['wandb_project'],
        config=args,
        job_type="CleanRepo",
        name=args['wandb_name'] if args['wandb_name'] else generate_wandb_name(args),
    )


def run_experiment(args):
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    if args['log_wandb']:
        init_wandb(args)

    print(args)

    if args['algorithm'] == 'DFFKE':
        data_distributor = DataDistributor(args)
        data_free_federated_knowledge_exchange(args, data_distributor)
    elif args['algorithm'] == 'DFRD':
        from baselines.DFRD.main_DFRD import DFRD
        data_distributor = DataDistributor(args)
        DFRD(args, data_distributor)
    else:
        run_baseline(args)


if __name__ == '__main__':
    args = init_args()
    run_experiment(args)
    print('Experiment Complete.')