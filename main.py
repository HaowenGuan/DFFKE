import os
import copy
import yaml
import wandb
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from DFFKE import data_free_federated_knowledge_exchange_with_latent_generator
from utils_train import general_one_epoch, embedding_test
from models.model_factory_fn import get_generator, init_client_nets
from dataset.transforms import get_cifar_transform, get_mini_image_transform, get_vit_original_size_transform
from dataset.utils_dataset import load_dataset, get_federated_learning_dataset, CustomDataset


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='path to the config file')
    args = parser.parse_args()

    current_folder = os.path.dirname(os.path.abspath(__file__))
    args.config_file = current_folder + "/configs/FC100.yaml"
    with open(args.config_file, 'r') as file:
        args = yaml.safe_load(file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['cuda_gpu'])

    if not torch.cuda.is_available():
        args['device'] = 'cpu'
    return args


def generate_wandb_name(args):
    name = ''
    if args['knowledge_exchange_rounds'] > 0:
        name += args['generator_model'] + ' / '
        name += 'DFFKE'
        if args['review_local_after_knowledge_exchange']:
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
    # Only use when continue from a previous run
    for _ in range(0):
        wandb.log({})


def save_checkpoint(args, clients, optimizers, checkpoint_folder):
    if args['save_clients']:
        folder = args['checkpoint_dir'] + checkpoint_folder
        mkdir(folder)
        file_name = f'{args["dataset"]}_{args["n_clients"]}client_{args["alpha"]}alpha_{args["client_encoder"]}_checkpoint'
        checkpoint = {}
        for c_id, client in clients.items():
            checkpoint[c_id] = {
                'model_state_dict': client.state_dict(),
                'optimizer_state_dict': optimizers[c_id].state_dict(),
            }
        # Save the checkpoint
        torch.save(checkpoint, folder + f'{file_name}.pt')
        print(f'>> Saved clients checkpoint to {folder + file_name}.pt')


def evaluate(clients, loader, dataset, name, mode='cls_test', device='cuda'):
    """
    Evaluate the performance of each client on the given full dataset
    @param clients: dict
    @param loader: DataLoader
    @param dataset: 'Train' or 'Test'
    @param name: Special Prefix, 'Local Aligned' or 'Global Exchanged'
    @param mode: 'emb_test' or 'cls_test'
    @param device: torch.device
    """
    print(f"Testing Each Client's Performance on {dataset} Set after {name}")
    results = ''
    client_loss_list = []
    client_acc_list = []
    client_acc_dict = {}
    for c_id, client in tqdm(clients.items()):
        client.eval()
        if mode == 'emb_test':
            client_loss, client_acc = embedding_test(client, loader, False, device)
        elif mode == 'cls_test':
            client_loss, client_acc = general_one_epoch(client, loader, None, device)
        else:
            raise ValueError('Unknown mode')
        results += f'{c_id}:({client_loss:.2f},{client_acc:.2f}) '
        client_loss_list.append(client_loss)
        client_acc_list.append(client_acc)
        client_acc_dict[c_id] = client_acc
    print(f">> {dataset} Set (Loss,Acc): {results}")
    print(f'>> Avg:({np.mean(client_loss_list):.2f},{np.mean(client_acc_list):.2f})')

    if args['log_wandb']:
        wandb.log({
            f'{name} {dataset} Set Loss': np.mean(client_loss_list),
            f'{name} {dataset} Set Acc': np.mean(client_acc_list)
        })
    return client_acc_dict


def pure_student_evaluation(pure_student, train_loader, test_loader, device):
    ps_train_cls_loss, ps_train_cls_acc = general_one_epoch(pure_student, train_loader, None, device)
    print(f'Pure Student train set cls Loss {ps_train_cls_loss:.3f}, Acc {ps_train_cls_acc:.3f}')
    ps_test_cls_loss, ps_test_cls_acc = general_one_epoch(pure_student, test_loader, None, device)
    print(f'Pure Student test set cls Loss {ps_test_cls_loss:.3f}, Acc {ps_test_cls_acc:.3f}')
    if args['log_wandb']:
        wandb.log({
            'Pure Student Train Set CLS Loss': ps_train_cls_loss,
            'Pure Student Train Set CLS Acc': ps_train_cls_acc,
            'Pure Student Test Set CLS Loss': ps_test_cls_loss,
            'Pure Student Test Set CLS Acc': ps_test_cls_acc,})


def run_experiment(args):
    import os
    # Set random seed
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    mkdir(args['checkpoint_dir'])

    if args['log_wandb']:
        init_wandb(args)

    print(args)
    device = args['device']
    print(f'>> Using device {args["device"]}')

    print(f'>> Split data for FL with Dir({args["alpha"]})')
    X_train_clients, Y_train_clients, X_test, Y_test, client_class_cnt = get_federated_learning_dataset(
        args['dataset'], args['data_dir'], args['n_clients'], alpha=args['alpha'], redo_split=args['redo_ds_split'])
    X_train, Y_train, _, _, n_class = load_dataset(args['dataset'], args['data_dir'])

    print(">> Initializing clients models")
    clients = init_client_nets(args['n_clients'] + 1, args['client_encoder'], n_class, device)
    pure_student = clients[args['n_clients']]
    del clients[args['n_clients']]
    client_optimizers = {}
    for client_id, client in clients.items():
        client_optimizers[client_id] = optim.Adam(client.parameters(), lr=args['client_lr'], weight_decay=args['reg'])

    client_train_transform = client_test_transform = None
    if args['client_encoder'] == 'resnet18':
        if args['dataset'] in ['CIFAR10', 'CIFAR100', 'FC100']:
            transform = get_cifar_transform()
            client_train_transform = transform['train_transform']
            client_test_transform = transform['test_transform']
        elif args['dataset'] == 'miniImageNet':
            transform = get_mini_image_transform()
            client_train_transform = transform['train_transform']
            client_test_transform = transform['test_transform']
    elif 'clip' in args['client_encoder']:
        transform = get_vit_original_size_transform()
        client_train_transform = transform['train_transform']
        client_test_transform = transform['test_transform']
    else:
        raise ValueError('Unknown encoder')

    train_loaders = {}
    fixed_train_loaders = {}
    for client_id, client in clients.items():
        # Get the private data for the client
        X_train_client = X_train_clients[client_id]
        Y_train_client = Y_train_clients[client_id]
        print(f'>> Client {client_id} owns {len(X_train_client)} training samples.')
        print(sorted(client_class_cnt[client_id], reverse=True))

        # Create the private data loader for each client
        train_dataset = CustomDataset(X_train_client, Y_train_client, transform=client_train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
        train_loaders[client_id] = train_loader

        # Create the fixed data loader for each client
        train_dataset = CustomDataset(X_train_client, Y_train_client, transform=client_test_transform)
        fixed_train_loaders[client_id] = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)

    # full training set data loader for evaluation purpose
    print(f'>> Full Train Set Size: {len(X_train)}')
    train_dataset = CustomDataset(X_train, Y_train, transform=client_test_transform)
    full_train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)

    print(f'>> Full Test Set Size: {len(X_test)}')
    test_dataset = CustomDataset(X_test, Y_test, transform=client_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)

    ######################################## Warmup Clients Model ########################################
    # Load the checkpoint if needed
    if args['load_clients'] is not None:
        file_name = f'{args["dataset"]}_{args["n_clients"]}client_{args["alpha"]}alpha_{args["client_encoder"]}_checkpoint'
        checkpoint_path = args['load_clients'] + file_name + '.pt'
        # Check if the checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f'>> Checkpoint file {checkpoint_path} does not exist. Skip loading clients.')
        else:
            print(f'>> Loading clients checkpoint from {checkpoint_path}')
            state_dict = torch.load(checkpoint_path)
            for c_id, client in clients.items():
                del state_dict[c_id]['model_state_dict']['docking.weight']
                del state_dict[c_id]['model_state_dict']['docking.bias']
                client.load_state_dict(state_dict[c_id]['model_state_dict'], strict=False)
                # client_optimizers[c_id].load_state_dict(state_dict[c_id]['optimizer_state_dict'])
            print(f'>> Loaded.')


    local_aligned_best_test_loss = {c_id: 0 for c_id in clients}
    local_aligned_best_test_acc = {c_id: 0 for c_id in clients}
    if args['warmup_clients']:
        # Initialize client optimizers
        print(">> Warmup Each Clients:")
        training_clients = {c_id: client for c_id, client in clients.items()}
        client_training_loss_acc = {}
        client_testing_loss_acc = {}
        epoch = 0
        while len(training_clients) > 0:
            epoch += 1
            train_results = ''
            test_results = ''
            for c_id, client in list(training_clients.items()):
                client.train()
                client_loss, client_acc = general_one_epoch(client, train_loaders[c_id], client_optimizers[c_id], device)
                client_training_loss_acc[c_id] = (client_loss, client_acc)
                if client_acc > args['local_acc']:
                    del training_clients[c_id]
                client.eval()
                client_loss, client_acc = general_one_epoch(client, test_loader, None, device)
                client_testing_loss_acc[c_id] = (client_loss, client_acc)
                if client_acc > local_aligned_best_test_acc[c_id]:
                    local_aligned_best_test_loss[c_id] = client_loss
                    local_aligned_best_test_acc[c_id] = client_acc
                    
            for k, loss_acc in client_training_loss_acc.items():
                train_results += f'{k}:({loss_acc[0]:.2f},{loss_acc[1]:.2f}) '
            print(f">> Epoch {epoch}, Client Training (Loss,Acc): {train_results[:-1]}")
            for k, loss_acc in client_testing_loss_acc.items():
                test_results += f'{k}:({loss_acc[0]:.2f},{loss_acc[1]:.2f}) '
            print(f">> Epoch {epoch}, Client Testing (Loss,Acc):  {test_results[:-1]}")
            if args['log_wandb']:
                wandb.log({'Local Aligned Best Test Set Loss': np.mean(list(local_aligned_best_test_loss.values())),
                           'Local Aligned Best Test Set Acc': np.mean(list(local_aligned_best_test_acc.values()))})

        save_checkpoint(args, clients, client_optimizers, checkpoint_folder='warmup/')
        print(">> Warmup Clients Finished.")

    evaluate(clients, full_train_loader, 'Train', 'Local Aligned', 'cls_test', device)
    evaluate(clients, test_loader, 'Test', 'Local Aligned', 'cls_test', device)
    pure_student_evaluation(pure_student, full_train_loader, test_loader, device)
    print("-------------------------------------------------------------------------------------------------")

    ######################################## Knowledge Distill ########################################
    local_aligned_best_test_loss = {c_id: 0 for c_id in clients}
    local_aligned_best_test_acc = {c_id: 0 for c_id in clients}
    data_banks = {c_id: None for c_id in clients}
    FKE_clients = {c_id: client for c_id, client in clients.items()}
    
    for round_i in range(args['knowledge_exchange_rounds']):
        print(f'>> Current Round: {round_i}')

        if args['new_client_opt_every_round']:
            # New optimizer for each client every round
            for client_id, client in clients.items():
                client_optimizers[client_id] = optim.Adam(client.parameters(), lr=args['client_lr'], weight_decay=args['reg'])

        generator = get_generator(model_name=args['generator_model'], nz=clients[0].output.in_features, n_cls=n_class)
        generator.to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args['generator_model_lr'])

        data_free_federated_knowledge_exchange_with_latent_generator(
            args=args,
            clients=clients,
            client_optimizers=client_optimizers,
            client_data_loaders=fixed_train_loaders,
            client_aug_data_loaders=train_loaders,
            client_class_cnt=client_class_cnt,
            generator=generator,
            generator_optimizer=generator_optimizer,
            batch_size=args['batch_size'],
            L2G_epoch=args['L2G_epoch'],
            G2L_epoch=args['G2L_epoch'],
            device=device,
            pure_student=pure_student,
            data_banks=data_banks,
            FKE_clients=FKE_clients,
        )
        FKE_clients = {c_id: copy.deepcopy(client) for c_id, client in clients.items()}

        pure_student_evaluation(pure_student, full_train_loader, test_loader, device)
        evaluate(clients, full_train_loader, 'Train', 'Global Exchanged', 'cls_test', device)
        evaluate(clients, test_loader, 'Test', 'Global Exchanged', 'cls_test', device)
        
        if args['review_local_after_knowledge_exchange']:
            training_clients = {c_id: client for c_id, client in clients.items()}
            client_training_loss_acc = {}
            while len(training_clients) > 0:
                train_results = ''
                for c_id, client in list(training_clients.items()):
                    client.train()
                    client_loss, client_acc = general_one_epoch(client, train_loaders[c_id], client_optimizers[c_id], device)
                    client_training_loss_acc[c_id] = (client_loss, client_acc)
                    if client_acc > args['local_acc']:
                        del training_clients[c_id]
                    client.eval()

                for c_id, loss_acc in client_training_loss_acc.items():
                    train_results += f'{c_id}:({loss_acc[0]:.2f},{loss_acc[1]:.2f}) '
                print(f">> Client Training (Loss,Acc): {train_results[:-1]}")
            
            evaluate(clients, full_train_loader, 'Train', 'Local Aligned', 'cls_test', device)
            evaluate(clients, test_loader, 'Test', 'Local Aligned', 'cls_test', device)
            
        print("-------------------------------------------------------------------------------------------------")

    save_checkpoint(args, clients, client_optimizers, checkpoint_folder='knowledge_exchange/')
    print("DFFKE Algorithm Ended.")


if __name__ == '__main__':
    args = init_args()
    run_experiment(args)