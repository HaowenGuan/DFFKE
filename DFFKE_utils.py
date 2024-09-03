import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def general_one_epoch(net, data_loader, optimizer=None, device='cpu'):
    """
    General one epoch function for training and validation
    [Note] if optimizer is provided, it will train the model.
    Make sure to call net.train() and net.eval() accordingly before calling this function.
    """
    total_loss = 0
    total_acc = 0
    data_num = 0
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
    return total_loss, total_acc


def train_output_layer(
        net,
        data_loader,
        output_layer,
        use_docking=False,
        lr=0.1,
        patient=5,
        device='cpu'):
    """
    Train the output layer of a network with fixed feature extractor
    """
    net.eval()
    emb_y = []
    data_num = 0
    for i, (data, target) in enumerate(data_loader):
        data_num += len(target)
        data, target = data.to(device), target.to(device)
        emb, _ = net(data, use_docking=use_docking)
        emb_y.append((emb.detach(), target))

    output_layer = output_layer.to(device)
    # With lower lr, the ACC can go as high as 0.32 for a Naive model on CIFAR-100 dataset
    # However, we want to efficiently evaluate embedding quality, getting high ACC is not the goal
    output_layer_optimizer = torch.optim.Adam(output_layer.parameters(), lr=lr)
    best_loss, best_acc, wait = float('inf'), -1, 0
    cur_losses, cur_accs = [], []
    # i = 0
    while True:
        # i += 1
        total_loss = total_acc = 0
        for emb, target in emb_y:
            output_layer.zero_grad()
            output = output_layer(emb)
            loss = F.cross_entropy(output, target)
            total_acc += (torch.argmax(output, dim=1) == target).float().sum().item()
            total_loss += loss.item() * len(target)
            loss.backward()
            output_layer_optimizer.step()
        total_loss /= data_num
        total_acc /= data_num

        if total_acc > best_acc:
            best_loss = total_loss
            best_acc = total_acc
            wait = 0
            cur_losses = [total_loss]
            cur_accs = [total_acc]
        else:
            wait += 1
            cur_losses.append(total_loss)
            cur_accs.append(total_acc)
            if wait == patient:
                break

    return np.mean(cur_losses), np.mean(cur_accs)


def embedding_test(net, data_loader, use_docking=False, device='cpu'):
    """
    Test the model's embedding output in classification task
    """
    output_layer = nn.Linear(net.output.in_features, net.output.out_features).to(device)
    # With lower lr, the ACC can go as high as 0.32 for a Naive model on CIFAR-100 dataset
    # However, we want to efficiently evaluate embedding quality, getting high ACC is not the goal
    loss, acc = train_output_layer(
        net=net,
        data_loader=data_loader,
        output_layer=output_layer,
        use_docking=use_docking,
        lr=0.1,
        patient=5,
        device=device
    )
    return loss, acc


def save_checkpoint(args, clients, optimizers, checkpoint_folder):
    if args['save_clients']:
        folder = args['checkpoint_dir'] + checkpoint_folder
        mkdir(folder)
        file_name = f'{args["dataset"]}_{args["n_clients"]}client_{args["alpha"]}alpha_{args["client_encoder"]}_checkpoint'
        checkpoint = {}
        for c_id, client in enumerate(clients):
            checkpoint[c_id] = {
                'model_state_dict': client.state_dict(),
                'optimizer_state_dict': optimizers[c_id].state_dict(),
            }
        # Save the checkpoint
        torch.save(checkpoint, folder + f'{file_name}.pt')
        print(f'>> Saved clients checkpoint to {folder + file_name}.pt')


def local_align_clients(clients, optimizers, train_loaders, passing_acc, test_loader, log_wandb=False, device='cpu'):
    training_clients = {c_id: client for c_id, client in enumerate(clients)}
    client_training_loss_acc = {}
    client_testing_loss_acc = {}
    epoch = 0
    while len(training_clients) > 0:
        epoch += 1
        train_results = ''
        test_results = ''
        for c_id, client in list(training_clients.items()):
            client.train()
            client_loss, client_acc = general_one_epoch(client, train_loaders[c_id], optimizers[c_id], device)
            client_training_loss_acc[c_id] = (client_loss, client_acc * 100)
            if client_acc > passing_acc:
                del training_clients[c_id]
            client.eval()
            if test_loader is not None:
                client_loss, client_acc = general_one_epoch(client, test_loader, None, device)
                client_testing_loss_acc[c_id] = (client_loss, client_acc * 100)

        for k, loss_acc in client_training_loss_acc.items():
            train_results += f'{k}:({loss_acc[0]:.2f},{loss_acc[1]:.1f}) '
        print(f">> Epoch {epoch}, Client Training (Loss,Acc): {train_results[:-1]}")

        if test_loader is not None:
            for k, loss_acc in client_testing_loss_acc.items():
                test_results += f'{k}:({loss_acc[0]:.2f},{loss_acc[1]:.1f}) '
            print(f">> Epoch {epoch}, Client Testing (Loss,Acc):  {test_results[:-1]}")
            if log_wandb:
                wandb.log({'Local Aligned Test Set Loss': np.mean([x[0] for x in client_testing_loss_acc.values()]),
                           'Local Aligned Test Set Acc': np.mean([x[1] for x in client_testing_loss_acc.values()])})


def evaluate(clients, loader, dataset, name, mode='cls_test', log_wandb=False, device='cpu'):
    """
    Evaluate the performance of each client on the given full dataset
    @param clients: dict
    @param loader: DataLoader
    @param dataset: 'Train' or 'Test'
    @param name: Special Prefix, 'Local Aligned' or 'Global Exchanged'
    @param mode: 'emb_test' or 'cls_test'
    @param log_wandb: bool
    @param device: torch.device
    """
    print(f"Testing Each Client's Performance on {dataset} Set after {name}")
    results = ''
    client_loss_list = []
    client_acc_list = []
    client_acc_dict = {}
    for c_id, client in tqdm(list(enumerate(clients))):
        client.eval()
        if mode == 'emb_test':
            client_loss, client_acc = embedding_test(client, loader, False, device)
        elif mode == 'cls_test':
            client_loss, client_acc = general_one_epoch(client, loader, None, device)
        else:
            raise ValueError('Unknown mode')
        results += f'{c_id}:({client_loss:.2f},{client_acc * 100:.1f}) '
        client_loss_list.append(client_loss)
        client_acc_list.append(client_acc * 100)
        client_acc_dict[c_id] = client_acc
    print(f">> {dataset} Set (Loss,Acc): {results}")
    print(f'>> Avg (Loss, Acc, Std):({np.mean(client_loss_list):.2f}, {np.mean(client_acc_list):.2f}, {np.std(client_acc_list):.2f})')

    if log_wandb:
        wandb.log({
            f'{name} {dataset} Set Loss': np.mean(client_loss_list),
            f'{name} {dataset} Set Acc': np.mean(client_acc_list)
        })
    return client_acc_dict


def pure_student_evaluation(pure_student, train_loader, test_loader, log_wandb=False, device='cpu'):
    ps_train_cls_loss, ps_train_cls_acc = general_one_epoch(pure_student, train_loader, None, device)
    print(f'Pure Student train set cls Loss {ps_train_cls_loss:.3f}, Acc {ps_train_cls_acc * 100:.2f}')
    ps_test_cls_loss, ps_test_cls_acc = general_one_epoch(pure_student, test_loader, None, device)
    print(f'Pure Student test set cls Loss {ps_test_cls_loss:.3f}, Acc {ps_test_cls_acc * 100:.2f}')
    if log_wandb:
        wandb.log({
            'Pure Student Train Set CLS Loss': ps_train_cls_loss,
            'Pure Student Train Set CLS Acc': ps_train_cls_acc * 100,
            'Pure Student Test Set CLS Loss': ps_test_cls_loss,
            'Pure Student Test Set CLS Acc': ps_test_cls_acc * 100,})
