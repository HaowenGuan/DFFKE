import os
import time
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import io


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
    data_partition = args['data_partition']
    return f'{ds}_{mf}_{n_clients}client_{alpha}alpha_{local_align_acc}acc_{data_partition}_checkpoint.pt'


def save_checkpoint(args, clients, optimizers, checkpoint_folder, pure_student=None):
    folder = str(os.path.join(args['checkpoint_dir'], checkpoint_folder))
    mkdir(folder)
    file_path = str(os.path.join(folder, get_checkpoint_file_name(args)))
    checkpoint = {}
    for c_id, client in clients:
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
        test_loaders=None,
        log_wandb=False,
        device='cpu'):
    """
    Align clients locally, each client will be trained until it reaches the passing accuracy on its private train data
    :param clients: list of clients
    :param optimizers: list of optimizers for each client
    :param train_loaders: list of each clients private train loaders
    :param passing_acc: threshold for passing accuracy
    :param test_loaders: list of test loaders. If provided, will be evaluated iteratively during training
    :param log_wandb: whether to log to wandb
    :param device: torch.device
    """
    print('=' * 32)
    if passing_acc <= 1.0:
        passing_acc *= 100

    # Select clients that need training
    training_clients = {}
    for c_id, client in clients:
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
            if test_loaders is not None:
                with torch.no_grad():
                    client_loss, client_acc = run_one_epoch(client, test_loaders[c_id], None, device)
                client_testing_loss_acc[c_id] = (client_loss, client_acc)

        if not test_loaders:
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
                wandb.log({
                    'Local Aligned Personalized Test Set Loss':
                        np.mean([x[0] for x in client_testing_loss_acc.values()]),
                    'Local Aligned Personalized Test Set Acc':
                        np.mean([x[1] for x in client_testing_loss_acc.values()])})
    print('=' * 32)


def evaluate(clients, loaders, dataset, name, log_wandb=False, device='cpu'):
    """
    Evaluate the performance of each client on the given full dataset
    :param clients: list of clients model
    :param loaders: list of dataset loaders used for evaluation for each client
    :param dataset: 'Train' or 'Test'
    :param name: Special name Prefix, such as 'Local Aligned' or 'Global Exchanged'
    :param log_wandb: whether to log to wandb
    :param device: torch.device
    :return: list of accuracy
    """
    if len(loaders) < len(clients):
        print(f'>> Warning: Not all clients have data loaders for evaluating: {name} {dataset}')
        return [0]
    print(f"Testing Each Client's Performance on {dataset} Set after {name}")
    results = ''
    loss_list = []
    acc_list = []
    for c_id, client in tqdm(clients):
        client.eval()
        with torch.no_grad():
            loss, acc = run_one_epoch(client, loaders[c_id], None, device)
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
            'Pure Student Test Set CLS Acc': ps_test_cls_acc, })


def approximate_model_sensitivity(clients, loaders, device='cpu'):
    """
    Approximate the model sensitivity of each client on the given train dataset.

    For differential privacy, we need to calculate the sensitivity of the model.
    The sensitivity is the maximum difference in the model output when the input data changes by exactly one variable.
    In this function, we randomly change one pixel of the input image and calculate the difference in the model output.

    :param clients: list of (client_id, client_model) pairs
    :param loaders: list of dataset loaders used for evaluation for each client
    :param device: torch.device
    :return: float value of the sensitivity
    """
    if len(loaders) < len(clients):
        print(f'>> Warning: Not all clients have data loaders')
        return 0

    print("Approximating Model Sensitivity for Differential Privacy")
    emd_sensitivity = []
    logit_sensitivity = []
    for c_id, client in tqdm(clients):
        client_max_emd_diff = client_max_logit_diff = 0
        client.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(loaders[c_id]):
                data = data.to(device)
                B, C, H, W = data.shape
                emb, logit = client(data)

                # Use vectorized indexing to change one random pixel for each image in the batch
                batch_indices = torch.arange(B, device=device)
                c_indices = torch.randint(0, C, (B,), device=device)
                h_indices = torch.randint(0, H, (B,), device=device)
                w_indices = torch.randint(0, W, (B,), device=device)
                data[batch_indices, c_indices, h_indices, w_indices] = torch.randn(B, device=device)
                emb_perturbed, logit_perturbed = client(data)

                # Compute the l2 norm difference for each sample in the batch
                emd_diff = (emb - emb_perturbed).view(B, -1).norm(p=2, dim=1)
                emd_batch_max = emd_diff.max().item()
                if emd_batch_max > client_max_emd_diff:
                    client_max_emd_diff = emd_batch_max

                logit_diff = (logit - logit_perturbed).view(B, -1).norm(p=2, dim=1)
                logit_batch_max = logit_diff.max().item()
                if logit_batch_max > client_max_logit_diff:
                    client_max_logit_diff = logit_batch_max

        emd_sensitivity.append(client_max_emd_diff)
        logit_sensitivity.append(client_max_logit_diff)

    return np.mean(emd_sensitivity), np.mean(logit_sensitivity)



def save_decoder_visualization(directory, round_i, clients, decoder, client_visual_loaders, n_class, device='cuda'):
    """
    Save the visualization of the decoder
    :param directory: directory path to save the visualization data
    :param round_i: current round number
    :param clients: list of clients' model
    :param decoder: decoder model
    :param client_visual_loaders: list of client's data loader for visualization
    :param n_class: number of classes
    :param device: torch.device
    """
    visual_folder = os.path.join(directory, 'synthetic_data_visualization')
    mkdir(visual_folder)
    visual_folder = os.path.join(visual_folder, f'round_{round_i}')
    mkdir(visual_folder)
    mean = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).to(device)
    std = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).to(device)
    for i, client in enumerate(clients):
        client_visual_folder = os.path.join(visual_folder, f'client_{i}')
        mkdir(client_visual_folder)
        visual_loader = client_visual_loaders[i]
        for x, y in visual_loader:
            x, y = x.to(device), y.to(device)
            emb, logit = client(x, use_docking=True)
            y_one_hot = torch.zeros((len(y), n_class)).to(device)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1)
            fake_data = decoder(emb.detach(), y_one_hot)[0]
            # Save to folder
            fake_data = fake_data * std[:, None, None] + mean[:, None, None]
            fake_data = (fake_data * 255).clamp(0, 255).byte()
            image = transforms.ToPILImage()(fake_data.permute(0, 1, 2))
            image.save(os.path.join(client_visual_folder, f'{i}_{y.item()}_{round_i}.png'))
    print(f'>> Visualized Decoder Output Saved to {visual_folder}')


def estimate_file_size(item):
    buf = io.BytesIO()
    torch.save(item, buf)
    size_bytes = buf.tell()
    size_mb = size_bytes / (1024**2)
    return size_mb
