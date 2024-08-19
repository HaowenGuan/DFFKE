from dataset.utils_dataset import *
from models.feature_extractor.fe_utils import DiffAugment
from DFFKE_losses import contrastive
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import wandb


def few_shot_prototype(net, x_sup, y_sup, x_qry, y_qry, use_docking=False):
    """
    Perform [prototype calculation] + [cosine similarity] for few-shot learning.
    https://arxiv.org/pdf/2003.04390
    Based on paper, meta training with this approach will overfit training base classes.
    Specifically, meta testing acc on base classes will increase but novel classes will decrease.
    [Warning]
    - Make sure to call net.train() and net.eval() accordingly before this function.
    - Make sure net and data are on the same device before this function.
    """
    embedding_sup, y_hat_sup = net(x_sup, use_docking=use_docking)
    embedding_qry, y_hat_qry = net(x_qry, use_docking=use_docking)
    classes = sorted(set(y_sup.tolist()))
    prototype = torch.zeros(len(classes), embedding_sup.size(1), device=embedding_sup.device)
    new_y_qry = torch.zeros_like(y_qry)
    for i, c in enumerate(classes):
        prototype[i] = embedding_sup[y_sup == c].mean(0)
        new_y_qry[y_qry == c] = i
    y_qry = new_y_qry
    prototype = F.normalize(prototype) # C x Z
    embedding_qry = F.normalize(embedding_qry) # Q x Z
    logits = torch.mm(embedding_qry, prototype.t())# * net.tau # Q x C
    loss = F.cross_entropy(logits, y_qry)
    acc = (logits.argmax(1) == y_qry).float().mean().item()
    return loss, acc


def few_shot_logistic_regression(net, x_sup, y_sup, x_qry, y_qry):
    """
    Perform logistic regression on the support set and predict the query set.
    """
    embedding_sup, _, _ = net(x_sup)
    embedding_qry, _, _ = net(x_qry)

    classes = sorted(set(y_sup.tolist()))
    new_y_sup = torch.zeros_like(y_sup)
    new_y_qry = torch.zeros_like(y_qry)
    for i, c in enumerate(classes):
        new_y_sup[y_sup == c] = i
        new_y_qry[y_qry == c] = i
    y_sup = new_y_sup
    y_qry = new_y_qry

    def l2_normalize(x):
        norm = (x.pow(2).sum(1, keepdim=True) + 1e-9).pow(1. / 2)
        out = x.div(norm + 1e-9)
        return out

    embedding_sup = l2_normalize(embedding_sup.detach().cpu()).numpy()
    embedding_qry = l2_normalize(embedding_qry.detach().cpu()).numpy()

    clf = LogisticRegression(penalty='l2',
                             random_state=0,
                             C=1.0,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')
    clf.fit(embedding_sup, y_sup.detach().cpu().numpy())

    # query_y_hat = clf.predict(embedding_qry)
    query_y_hat_prob = torch.tensor(clf.predict_proba(embedding_qry)).to(y_qry.device)

    acc = (torch.argmax(query_y_hat_prob, -1) == y_qry).float().mean().item()
    return acc

def meta_train_net(args, net, optimizer, x, y, transform, device='cpu'):
    """
    Train a network on a given dataset with meta learning
    :param args: arguments
    :param net: the network to train
    :param x: the training data for this client
    :param y: the training labels for this client
    :param transform: the transformation to apply to the data
    :param device: the device to use
    @param args: arguments
    @param net: the model to train
    @param optimizer: the optimizer of net
    @param x: the training data for this client
    @param y: the training labels for this client
    @param transform: transformation to apply to the data
    @param device: cpu or cuda
    @return:
    """
    N = args['meta_config']['train_client_class']
    K = args['meta_config']['train_support_num']
    Q = args['meta_config']['train_query_num']

    if args['dataset'] == 'FC100':
        class_dict = fine_split['train']
    else:
        raise ValueError('Unknown dataset')

    x_sup, y_sup, x_qry, y_qry = sample_few_shot_data(x, y, class_dict, transform, N, K, Q, device=device)
    # x_total = torch.cat([x_sup, x_qry], 0).to(device)
    # y_total = torch.cat([y_sup, y_qry], 0).long().to(device)

    # Meta Training
    ############################
    net.train()
    optimizer.zero_grad()
    loss, acc = few_shot_prototype(net, x_sup, y_sup, x_qry, y_qry)
    loss.backward()
    optimizer.step()

    return acc


def meta_test_net(args, net, X_test, Y_test, transform, ft_approach='prototype', test_k=None, device='cpu'):

    """
        test a net on a given dataset with meta learning
        @param args: arguments
        @param net: network to test
        @param X_test: test data
        @param Y_test: test label
        @param transform: transform to apply to the data
        @param ft_approach: few shot testing approach, 'prototype' or 'classic'
        @param test_k: specific k for the test, if None, use the default test_k
        @param device: cpu or cuda
        @return:
    """
    N = args['meta_config']['test_client_class']
    K = test_k if test_k else args['meta_config']['test_support_num']
    Q = args['meta_config']['test_query_num']

    if args['dataset'] == 'FC100':
        class_dict = fine_split['test']
    elif args['dataset'] == 'miniImageNet':
        class_dict = list(range(20))
    elif args['dataset'] == '20newsgroup':
        class_dict = [0, 2, 3, 8, 9, 15, 19]
    elif args['dataset'] == 'fewrel':
        class_dict = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]
    elif args['dataset'] == 'huffpost':
        class_dict = list(range(25, 41))
    else:
        raise ValueError('Unknown dataset')

    x_sup, y_sup, x_qry, y_qry = sample_few_shot_data(X_test, Y_test, class_dict, transform, N, K, Q, device=device)

    # Fine-tune with meta-test
    CELoss = nn.CrossEntropyLoss()
    test_net = copy.deepcopy(net).to(device)
    test_net.train()
    meta_config = args['meta_config']
    optimizer = optim.Adam(test_net.parameters(), lr=meta_config['test_ft_lr'], weight_decay=args['reg'])
    test_accs = []
    test_losses = []
    use_docking = args['meta_config']['test_use_docking']
    for step in range(meta_config['test_ft_steps']):
        if step % 10 == 0:
            test_net.eval()
            with torch.no_grad():
                loss, acc = few_shot_prototype(test_net, x_sup, y_sup, x_qry, y_qry, use_docking)
            test_accs.append(acc)
            test_losses.append(loss.item())
            test_net.train()
        optimizer.zero_grad()
        if ft_approach == 'prototype':
            aug_x_sup = DiffAugment(x_sup, meta_config['aug_types'], meta_config['aug_prob'], detach=True)
            loss, acc = few_shot_prototype(test_net, x_sup, y_sup, aug_x_sup, y_sup, use_docking)
        elif ft_approach == 'contrastive':
            embedding, _ = test_net(x_sup, use_docking=use_docking)
            loss = contrastive(embedding, embedding.detach(), y_sup, y_sup)
        else:  # classic
            _, logits = test_net(x_sup)
            loss = CELoss(F.softmax(logits, dim=1), y_sup)
        loss.backward()
        optimizer.step()
    # Final meta-test
    test_net.eval()
    with torch.no_grad():
        loss, acc = few_shot_prototype(test_net, x_sup, y_sup, x_qry, y_qry, use_docking)
    test_accs.append(acc)
    test_losses.append(loss.item())
    # acc = few_shot_logistic_regression(net, x_sup, y_sup, x_qry, y_qry)

    return test_accs, test_losses


def sample_few_shot_data(x, y, class_choices, img_transform, N, K, Q, device='cpu'):
    # Pick N picked_class. Make sure that there are at least K + Q samples for each class
    class_choices = class_choices.copy()
    current_min_size = 0
    picked_class = picked_class_data = []
    while current_min_size < K + Q:
        picked_class_data = []
        picked_class = np.random.choice(class_choices, N, replace=False).tolist()
        for i in picked_class:
            picked_class_data.append(x[y == i])
            if picked_class_data[-1].shape[0] < K + Q:
                class_choices.remove(i)
        current_min_size = min([one.shape[0] for one in picked_class_data])

    x_sup, y_sup = [], []
    x_qry, y_qry = [], []
    # sample K + Q samples for each picked class
    for class_index, class_data in zip(picked_class, picked_class_data):
        sample_idx = np.random.choice(list(range(class_data.shape[0])), K + Q, replace=False).tolist()
        x_sup.append(class_data[sample_idx[:K]])
        x_qry.append(class_data[sample_idx[K:]])
        y_sup.append(torch.ones(K) * class_index)
        y_qry.append(torch.ones(Q) * class_index)

    x_sup = np.concatenate(x_sup, 0)
    x_qry = np.concatenate(x_qry, 0)
    y_sup = torch.cat(y_sup, 0).long().to(device)
    y_qry = torch.cat(y_qry, 0).long().to(device)

    # Apply the same image transformation to the support set and the query set
    transformed_x_sup = []
    for i in range(x_sup.shape[0]):
        transformed_x_sup.append(img_transform(x_sup[i]))
    transformed_x_qry = []
    for i in range(x_qry.shape[0]):
        transformed_x_qry.append(img_transform(x_qry[i]))

    # Finalized meta-learning data
    x_sup = torch.tensor(torch.stack(transformed_x_sup, 0)).to(device)
    x_qry = torch.tensor(torch.stack(transformed_x_qry, 0)).to(device)

    return x_sup, y_sup, x_qry, y_qry


def clients_meta_train(args, clients, client_optimizers, X_train_clients, Y_train_clients, train_transform, device="cpu"):
    meta_train_acc = []

    print(f'>> Meta training {len(clients)} clients, {args["meta_config"]["num_train_tasks"]} tasks each')
    for net_id, client in tqdm(clients.items()):
        acc_train = []
        for _ in range(args['meta_config']['num_train_tasks']):
            acc_train.append(meta_train_net(args, client, client_optimizers[net_id], X_train_clients[net_id], Y_train_clients[net_id], train_transform, device=device))
        meta_train_acc.append(np.mean(acc_train))

    print('Meta Train Accuracy: ' + ' | '.join(['{:.4f}'.format(acc) for acc in meta_train_acc]))

    if args['log_wandb']:
        wandb.log({f'Clients Meta Train Accuracy': np.mean(meta_train_acc)})


def clients_meta_test(args, clients, X_test, Y_test, test_transform, device="cpu", log=True):
    total_test_acc = []
    total_test_loss = []

    print(f'>> Meta testing {len(clients)} clients, {args["meta_config"]["num_test_tasks"]} tasks each')
    for net_id, client in tqdm(clients.items()):
        test_acc = []
        test_loss = []
        for _ in range(args['meta_config']['num_test_tasks']):
            acc_list, loss_list = meta_test_net(
                args=args,
                net=client,
                X_test=X_test,
                Y_test=Y_test,
                transform=test_transform,
                ft_approach=args['meta_config']['test_ft_approach'],
                device=device,
            )
            test_acc.append(acc_list)
            test_loss.append(loss_list)
        total_test_acc.append(np.mean(test_acc, axis=0))
        total_test_loss.append(np.mean(test_loss, axis=0))

    total_test_acc = np.array(total_test_acc)
    total_test_loss = np.array(total_test_loss)
    for i in range(total_test_acc.shape[1]):
        print(f'Meta Test Accuracy at FT step {i*10}: ' + ' | '.join([f'{acc:.4f}' for acc in total_test_acc[:, i]]))

    if args['log_wandb'] and log:
        wandb_step = wandb.run.step
        for i in range(total_test_acc.shape[1]):
            wandb.log({f'Clients Meta Test Accuracy': np.mean(total_test_acc[:, i]),
                       f'Clients Meta Test Loss': np.mean(total_test_loss[:, i])}, step=wandb_step + i)