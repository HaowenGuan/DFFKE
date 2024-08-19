import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

        # print(f'Iter {i}: loss={total_loss:.4f}, acc={total_acc:.4f}')
        if total_acc > best_acc:
        # if total_loss < best_loss:
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
    # # With lower lr, the ACC can go as high as 0.32 for a Naive model on CIFAR-100 dataset
    # # However, we want to efficiently evaluate embedding quality, getting high ACC is not the goal
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
