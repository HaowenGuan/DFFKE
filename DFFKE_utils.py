import os
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def run_one_epoch(net, data_loader, optimizer=None, device='cpu'):
    """
    Run one epoch on given data loader for training or validation.
    !!! If optimizer is provided, it will train the model. Vice versa.
    !!! Make sure to call net.train() or net.eval() accordingly before calling this function.
    :param net: pytorch model
    :param data_loader: the data loader
    :param optimizer: If None, it will not train the model. Vice versa.
    :param device: torch.device
    :return: loss, accuracy (in percentage)
    """
    total_loss = total_acc = data_num = 0
    for i, (data, target) in enumerate(data_loader):
        data_num += len(target)
        data, target = data.to(device), target.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        h, y_hat = net(data)
        loss = F.cross_entropy(y_hat, target)
        acc = (torch.argmax(y_hat, dim=1) == target).sum().item()
        total_loss += loss.item() * len(target)
        total_acc += acc
        if optimizer is not None:
            loss.backward()
            optimizer.step()
    total_loss /= data_num
    total_acc /= data_num
    return total_loss, total_acc * 100


def get_checkpoint_file_name(args):
    ds = args['dataset']
    mf = args['model_family']
    n_clients = args['n_clients']
    alpha = float(args['alpha'])
    local_align_acc = int(args['local_align_acc'] * 100)
    client_encoder = args['client_encoder']
    return f'{ds}_{mf}_{n_clients}client_{alpha}alpha_{local_align_acc}acc_{client_encoder}_checkpoint.pt'


def save_checkpoint(args, clients, optimizers, checkpoint_folder, pure_student=None):
    if args['save_clients']:
        folder = os.path.join(args['checkpoint_dir'], checkpoint_folder)
        mkdir(folder)
        file_path = os.path.join(folder, get_checkpoint_file_name(args))
        checkpoint = {}
        for c_id, client in enumerate(clients):
            checkpoint[c_id] = {
                'model_state_dict': client.state_dict(),
                'optimizer_state_dict': optimizers[c_id].state_dict(),
            }
        if pure_student is not None:
            checkpoint['pure_student'] = {
                'model_state_dict': pure_student.state_dict(),
            }
        # Save the checkpoint
        torch.save(checkpoint, file_path)
        print(f'>> Saved clients checkpoint to {file_path}')


def local_align_clients(
        clients,
        optimizers,
        train_loaders,
        passing_acc,
        test_loader=None,
        log_wandb=False,
        device='cpu'):
    """
    Align clients locally, each client will be trained until it reaches the passing accuracy on its private train data
    :param clients: list of clients
    :param optimizers: list of optimizers for each client
    :param train_loaders: list of each clients private train loaders
    :param passing_acc: threshold for passing accuracy
    :param test_loader: global test loader. If provided, will be evaluated iteratively during training
    :param log_wandb: whether to log to wandb
    :param device: torch.device
    """
    print('=' * 32)
    if passing_acc <= 1.0: passing_acc *= 100

    # Select clients that need training
    training_clients = {}
    for c_id, client in enumerate(clients):
        client.eval()
        with torch.no_grad():
            client_loss, client_acc = run_one_epoch(client, train_loaders[c_id], None, device)
        if client_acc < passing_acc:
            training_clients[c_id] = client
        else:
            print(f'>> Client {c_id} passed with local acc {client_acc:.2f}%')

    print('>> Local Aligning clients...')
    client_training_loss_acc = {}
    client_testing_loss_acc = {}
    epoch = 0
    while len(training_clients) > 0:
        epoch += 1
        for c_id, client in list(training_clients.items()):
            client.train()
            client_loss, client_acc = run_one_epoch(client, train_loaders[c_id], optimizers[c_id], device)
            client_training_loss_acc[c_id] = (client_loss, client_acc)
            if client_acc > passing_acc:
                print(f'>> Client {c_id} passed with local acc {client_acc:.2f}%')
                del training_clients[c_id]
            client.eval()
            if test_loader is not None:
                with torch.no_grad():
                    client_loss, client_acc = run_one_epoch(client, test_loader, None, device)
                client_testing_loss_acc[c_id] = (client_loss, client_acc)

        if not test_loader:
            train_results = ''
            for k, loss_acc in client_training_loss_acc.items():
                train_results += f'{k}:({loss_acc[0]:.2f},{loss_acc[1]:.2f}) '
            print(f">> Epoch {epoch}, Client Training (Loss,Acc): {train_results[:-1]}")
        else:
            train_test_results = ''
            for k, loss_acc in client_training_loss_acc.items():
                train_test_results += f'{k}:({loss_acc[1]:.2f},{client_testing_loss_acc[k][1]:.2f}) '
            print(f">> Epoch {epoch}, Client Local (Train,Test) Acc: {train_test_results[:-1]}")
            if log_wandb:
                wandb.log({'Local Aligned Test Set Loss': np.mean([x[0] for x in client_testing_loss_acc.values()]),
                           'Local Aligned Test Set Acc': np.mean([x[1] for x in client_testing_loss_acc.values()])})
    print('=' * 32)


def evaluate(clients, loader, dataset, name, log_wandb=False, device='cpu'):
    """
    Evaluate the performance of each client on the given full dataset
    :param clients: list of clients model
    :param loader: dataset loader used for evaluation
    :param dataset: 'Train' or 'Test'
    :param name: Special name Prefix, such as 'Local Aligned' or 'Global Exchanged'
    :param log_wandb: whether to log to wandb
    :param device: torch.device
    :return: list of accuracy
    """
    print(f"Testing Each Client's Performance on {dataset} Set after {name}")
    results = ''
    loss_list = []
    acc_list = []
    for c_id, client in tqdm(list(enumerate(clients))):
        client.eval()
        with torch.no_grad():
            loss, acc = run_one_epoch(client, loader, None, device)
        results += f'{c_id}:({loss:.2f},{acc:.1f}) '
        loss_list.append(loss)
        acc_list.append(acc)
    print(f">> {dataset} Set (Loss,Acc): {results}")
    print(f'>> Avg (Loss, Acc, Std): ({np.mean(loss_list):.2f}, {np.mean(acc_list):.2f}, {np.std(acc_list):.2f})')

    if log_wandb:
        wandb.log({
            f'{name} {dataset} Set Loss': np.mean(loss_list),
            f'{name} {dataset} Set Acc': np.mean(acc_list)
        })
    return acc_list


def pure_student_evaluation(pure_student, train_loader, test_loader, log_wandb=False, device='cpu'):
    """
    Evaluate the performance of the pure student model
    :param pure_student: single pure student model
    :param train_loader: train set loader for evaluation
    :param test_loader: test set loader for evaluation
    :param log_wandb: whether to log to wandb
    :param device: torch.device
    """
    pure_student.eval()
    with torch.no_grad():
        ps_train_cls_loss, ps_train_cls_acc = run_one_epoch(pure_student, train_loader, None, device)
        ps_test_cls_loss, ps_test_cls_acc = run_one_epoch(pure_student, test_loader, None, device)
    print(f'Pure Student train set cls Loss {ps_train_cls_loss:.3f}, Acc {ps_train_cls_acc:.2f}')
    print(f'Pure Student test set cls Loss {ps_test_cls_loss:.3f}, Acc {ps_test_cls_acc:.2f}')
    if log_wandb:
        wandb.log({
            'Pure Student Train Set CLS Loss': ps_train_cls_loss,
            'Pure Student Train Set CLS Acc': ps_train_cls_acc,
            'Pure Student Test Set CLS Loss': ps_test_cls_loss,
            'Pure Student Test Set CLS Acc': ps_test_cls_acc,})
