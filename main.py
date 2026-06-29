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
    parser.add_argument('--device_id', type=str, default=None, help='the gpu id. This overwrites the config')
    run_args = parser.parse_args()

    current_folder = os.path.dirname(os.path.abspath(__file__))
    if run_args.group_experiment:
        main_config_file = f'{current_folder}/configs/{run_args.group_experiment}/__init__.yaml'
        # Load the main config file, the general settings of the group experiment
        with open(main_config_file, 'r') as file:
            args = yaml.safe_load(file)
        config_file = f'{current_folder}/configs/{run_args.group_experiment}/{run_args.config_file}'
        # Load the algorithm config file, overwrite the duplicated settings in the main config file
        with open(config_file, 'r') as file:
            sub_args = yaml.safe_load(file)
        for key, value in sub_args.items():
            args[key] = sub_args[key]
    else:
        # Load a standalone config file
        config_file = current_folder + f'/configs/{run_args.config_file}'
        with open(config_file, 'r') as file:
            args = yaml.safe_load(file)

    # Overwrite the device id if it is provided in the command line
    if run_args.device_id is not None:
        args['device_id'] = run_args.device_id

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device_id'])
    if not torch.cuda.is_available():
        args['device'] = 'cpu'
        print('>> CUDA is not available. Using CPU.')
    else:
        print(f'>> Using CUDA:{args["device_id"]}')

    return args


def generate_wandb_name(args):
    name = f'{args["algorithm"]} / '
    if args["algorithm"] == 'DFFKE' and args['knowledge_exchange_rounds'] > 0:
        name += args['decoder_model'] + ' / '
        name += 'DFFKE / '
        name += f'Dec {args["decoder_train_epoch"]}.fid / '
        name += f'FKE {args["FKE_epoch"]}'
        if args['memory_buffer_iteration'] > 0:
            name += f'.mem.'
        name += 'emb.kl / '
    name += f'{args["n_clients"]} Clients / '
    name += f'{args["local_align_acc"] * 100:.1f} {args["dataset"]} Dir({args["alpha"]}) {args["model_family"]}'
    if 'personalized' in args['data_partition']:
        name += ' PFL'
    if 'additive_gaussian_DP'in args and args['additive_gaussian_DP']:
        name += ' DP ' + str(float(args['additive_gaussian_DP']))
    return name


def init_wandb(args):
    wandb.init(
        sync_tensorboard=False,
        project=args['wandb_project'],
        config=args,
        job_type="CleanRepo",
        name=args['wandb_name'] if args['wandb_name'] else generate_wandb_name(args),
    )


def get_model_family(args):
    if 'models' in args:
        print('model choice list already exists:', args['models'])
        return

    if args['model_family'] == "HtFE1":
        args['models'] = [
            'resnet18()',
        ]
    elif args['model_family'] == "HtFE2":
        args['models'] = [
            'resnet18()',
            'mobilenet_v3_large()',
        ]
    elif args['model_family'] == "HtFE5":
        args['models'] = [
            'resnet18()',
            'mobilenet_v3_small()',
            'mobilenet_v3_large()',
            'shufflenet_v2_x1_5()',
            'shufflenet_v2_x2_0()',
        ]
    elif args['model_family'] == "HtFE10":
        args['models'] = [
            'resnet18()',  #
            'resnet34()',  #
            'resnet50()',  #
            'googlenet(aux_logits=False)',  #
            'efficientnet_v2_s()',  #
            'mobilenet_v3_small()',  # 67.668%, 41s/123s
            'mobilenet_v3_large()',  # 75.274%, 42s/112s
            'shufflenet_v2_x1_5()',  # 72.052%, 54s/140s
            'shufflenet_v2_x2_0()',  # 75.354%, 46s/120s
            f'vit_tiny_torch(image_size={args["resize_img"]})'  #
        ]
    elif args['model_family'] == "FedGH_CNN":
        args['models'] = [
            'FedGH_CNN()'
        ]
    elif args['model_family'] == "FedTGP_HtFE8":
        args['models'] = [
            'FedAvgCNN(in_features=3, dim=1600)',  # for 32x32 img
            'googlenet(aux_logits=False)',
            'mobilenet_v2()',
            'resnet18()',
            'resnet34()',
            'resnet50()',
            'resnet101()',
            'resnet152()'
        ]
    else:
        raise f'Unknown model family: {args["model_family"]}'

    if 'CIFAR' in args['dataset'].upper():
        for i in range(len(args['models'])):
            if 'resnet' in args['models'][i]:
                args['models'][i] = args['models'][i].replace('()', f'_32x32(image_size={args["resize_img"]})')

    print(f'Using model family: {args["model_family"]}, with models: {args["models"]}')


def run_experiment(args):
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    if args['log_wandb']:
        init_wandb(args)

    # get model choice list based on given family name
    get_model_family(args)

    print(args)

    data_distributor = DataDistributor(args) if args['use_data_distributor'] else None

    if args['algorithm'] == 'DFFKE':
        data_free_federated_knowledge_exchange(args, data_distributor)
    # elif args['algorithm'] == 'DFRD':
    #     from data_free_baselines.DFRD.main_DFRD import DFRD
    #     DFRD(args, data_distributor)
    else:
        run_baseline(args, data_distributor)


if __name__ == '__main__':
    # exit(0)
    args = init_args()
    run_experiment(args)
    print('Experiment Complete.')
