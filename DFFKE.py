import os
import copy
import wandb
import numpy as np
from tqdm import tqdm
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from DFFKE_losses import contrastive, inverse_cross_entropy
from models.model_factory_fn import get_generator, init_client_nets
from dataset.utils_dataset import EmbLogitSet, FakeDataset, InfiniteDataLoader, CustomDataset
from DFFKE_utils import mkdir, get_checkpoint_file_name, save_checkpoint, local_align_clients, evaluate, pure_student_evaluation


def generate_labels(n, class_num):
    """
    Generate labels for generating data.
    The labels are generated according to the proportion of existence of each class in the training set.

    @param n: int, number of samples to generate.
    @param class_num: np.array (num_classes,), number of samples in each class.
    @return: np.array (n,), generated labels.
    """
    labels = np.arange(n)
    proportions = class_num / class_num.sum()
    proportions = (np.cumsum(proportions) * n).astype(int)[:-1]
    labels_split = np.split(labels, proportions)
    for i in range(len(labels_split)):
        labels_split[i].fill(i)
    labels = np.concatenate(labels_split)
    np.random.shuffle(labels)
    return labels.astype(int)


def generate_soft_labels(n, class_num):
    """
    Generate soft labels for generating data.

    @param n: int, number of samples to generate.
    @param class_num: np.array (num_classes,), number of samples in each class.
    @return: np.array (n, num_classes), generated soft labels.
    """
    classes = class_num.shape[0]
    soft_labels = np.zeros((n, classes))
    # Get non-zero sample classes from clients
    non_zero_classes = np.where(class_num > 0)[0]
    for i in range(n):
        # Randomly shuffle the training classes
        sequence = np.random.permutation(non_zero_classes)
        remaining = 1.0
        # Randomly assign the proportion of remaining logits to each class
        for j in range(len(sequence)):
            if remaining < 0.001:
                soft_labels[i, sequence[j]] = remaining
                break
            v = np.random.uniform(0.9, 1.0) * remaining
            soft_labels[i, sequence[j]] = v
            remaining -= v

        # if np.random.rand() < 0.5:
        #     # Randomly shuffle the training classes
        #     sequence = np.random.permutation(non_zero_classes)
        #     remaining = 1.0
        #     # Randomly assign the proportion of remaining logits to each class
        #     for j in range(len(sequence) - 1):
        #         v = np.random.uniform(0, remaining)
        #         soft_labels[i, sequence[j]] = v
        #         remaining -= v
        #     soft_labels[i, sequence[-1]] = remaining
        # else:
        #     # Assign a more confident label
        #     soft_labels[i, np.random.choice(non_zero_classes)] = 1.0
    return soft_labels


def get_batch_weight(labels, class_client_weight):
    """
    Compute the weight of each client contributing to the sample.
    Labels are in integer format.

    @param labels: np.array (bs,)
        each value [0, num_classes) is a integer class number.
    @param class_client_weight: np.array (num_classes, num_clients)
        each value [0, 1] is the proportion of the class in the client.
    @return: np.array (bs, num_clients)
    each value [0, 1] is the weight of clients contribute to the specific label.
    """
    bs = labels.size
    num_clients = class_client_weight.shape[1]
    batch_weight = np.zeros((bs, num_clients))
    batch_weight[np.arange(bs), :] = class_client_weight[labels, :]
    return batch_weight


def get_soft_batch_weight(soft_labels, class_client_weight):
    """
    Compute the weight of each client contributing to the sample.
    Labels are in probability format.

    @param soft_labels: np.array (bs, num_classes)
        each value [0, 1] is a likelihood of sample belonging to the class.
    @param class_client_weight: np.array (num_classes, num_clients)
        each value [0, 1] is the proportion of the class in the client.
    @return: np.array (bs, num_clients)
        each value [0, 1] is the weight of each client contributing to the sample.
    """
    bs = soft_labels.shape[0]
    num_clients = class_client_weight.shape[1]
    batch_weight = np.zeros((bs, num_clients))
    for i in range(num_clients):
        batch_weight[:, i] = np.sum(soft_labels * class_client_weight[:, i].reshape(1, -1), axis=1)
    return batch_weight


def local_to_generator(
        args,
        clients,
        client_data_loaders,
        client_aug_data_loaders,
        client_class_cnt,
        generator,
        generator_optimizer,
        batch_size,
        L2G_epoch,
        pure_student,
        knowledge_exchanged_clients,
        device='cuda',):

    num_clients, num_classes = client_class_cnt.shape
    L2G_client_data_loader = client_aug_data_loaders if args['L2G_augment_logits'] else client_data_loaders

    ###################################### Clustering Clients Data Distribution ######################################

    for client in clients: client.eval() # Freeze BN to collect client data embeddings
    wandb_step = wandb.run.step if args['log_wandb'] else 0

    # Collect data embeddings from clients
    client_emb_logit_loaders = {}
    n_samples = {}
    for c_id, client in enumerate(clients):
        client_emb = []
        client_logit = []
        client_target = []
        with torch.no_grad():
            for data, target in L2G_client_data_loader[c_id]:
                data, target = data.to(device), target.to(device)
                emb, logit = client(data, use_docking=False)
                client_emb.append(emb.detach())
                client_logit.append(F.softmax(logit, dim=1).detach())
                client_target.append(target)
        client_emb = torch.cat(client_emb, dim=0).cpu()
        client_logit = torch.cat(client_logit, dim=0).cpu()
        client_target = torch.cat(client_target, dim=0).cpu()
        emb_logit_set = EmbLogitSet(client_emb, client_logit, client_target)
        client_emb_logit_loaders[c_id] = DataLoader(emb_logit_set, batch_size=batch_size)
        n_samples[c_id] = len(client_target)

    # Train the docking layer for each client to optimally cluster the data embeddings
    cls_layer = nn.Linear(clients[0].docking.out_features, clients[0].output.out_features).to(device)
    clustering_params = list(cls_layer.parameters())
    for c_id, client in enumerate(clients):
        clustering_params.extend(client.docking.parameters())
    docking_optimizer = torch.optim.Adam(clustering_params, lr=0.001, weight_decay=1e-3)
    for s in tqdm(range(100)):
        clustering_loss = []
        for c_id, client in enumerate(clients):
            emb_logit_loader = client_emb_logit_loaders[c_id]
            loss = torch.Tensor([0]).to(device)
            docking_optimizer.zero_grad()
            for emb, _, target in emb_logit_loader:
                emb, target = emb.to(device), target.to(device)
                logit = cls_layer(client.docking(emb))
                loss += F.cross_entropy(logit, target, reduction='sum')
            loss /= n_samples[c_id]
            loss.backward()
            docking_optimizer.step()
            clustering_loss.append(loss.item())
        if args['log_wandb']:
            wandb.log({'Clustering Loss': np.mean(clustering_loss)}, step=wandb_step + s)

    # Collect the embeddings and logits of the clients
    client_emb_logit_loaders = []
    for c_id, client in enumerate(clients):
        client_emb = []
        client_logit = []
        client_target = []
        with torch.no_grad():
            for data, target in L2G_client_data_loader[c_id]:
                data, target = data.to(device), target.to(device)
                emb, logit = client(data, use_docking=True)
                client_emb.append(emb.detach())
                client_logit.append(F.softmax(logit, dim=1).detach())
                client_target.append(target)
        client_emb = torch.cat(client_emb, dim=0).cpu()
        client_logit = torch.cat(client_logit, dim=0).cpu()
        client_target = torch.cat(client_target, dim=0).cpu()
        # Determine the number of samples to keep (for example 10%)
        num_samples = int(args['L2G_sample_ratio'] * client_emb.size(0))
        indices = torch.randperm(client_emb.size(0))[:num_samples]
        emb_logit_set = EmbLogitSet(client_emb[indices], client_logit[indices], client_target[indices])
        client_emb_logit_loaders.append(DataLoader(emb_logit_set, batch_size=batch_size, shuffle=True, drop_last=True))

    ################################################ Training Generator ###############################################

    wandb_step = wandb.run.step if args['log_wandb'] else 0
    generator.train()
    for client in clients: client.eval()  # Freeze BN to distill original training knowledge
    for client in knowledge_exchanged_clients: client.eval()  # Freeze BN to contrast previous exchanged knowledge
    max_data_len = max([len(loader) for loader in client_emb_logit_loaders])
    L2G_iteration = L2G_epoch * max_data_len
    for s in tqdm(range(L2G_iteration)):
        # Direct sample generator
        inf_emb_logit_loaders = [InfiniteDataLoader(loader) for c_id, loader in enumerate(client_emb_logit_loaders)]
        emb_loss_list = []
        cls_loss_list = []
        md_loss_list = []
        for t_id, teacher in enumerate(clients):
            c_real_emb, c_real_logit, target = inf_emb_logit_loaders[t_id].get_next()
            c_real_emb, c_real_logit, target = c_real_emb.to(device), c_real_logit.to(device), target.to(device)

            generator_optimizer.zero_grad()
            y_one_hot = torch.zeros((len(target), num_classes)).to(device)
            y_one_hot.scatter_(1, target.unsqueeze(1), 1)
            fake_data = generator(c_real_emb, y_one_hot)
            c_fake_emb, c_fake_logit = teacher(fake_data, use_docking=True)
            total_loss = torch.Tensor([0]).to(device)
            # Knowledge Distillation Loss
            kd_loss = F.mse_loss(c_fake_emb, c_real_emb)
            emb_loss_list.append(kd_loss.item())
            total_loss += kd_loss
            # Classification Loss
            cls_loss = F.kl_div(F.log_softmax(c_fake_logit, dim=1), c_real_logit, reduction='batchmean')
            cls_loss_list.append(cls_loss.item())
            total_loss += cls_loss
            # Model Discrepancy Loss
            if args['L2G_use_emb_md_loss']:
                all_fake_emb = [c_fake_emb]
                for s_id, student in knowledge_exchanged_clients.items():
                    if s_id == t_id:
                        continue
                    s_fake_emb, _ = student(fake_data, use_docking=True)
                    all_fake_emb.append(s_fake_emb)
                all_fake_emb = torch.stack(all_fake_emb, dim=1)
                c_fake_emb = c_fake_emb.unsqueeze(1).detach()
                a_label = torch.ones(1).to(device)
                b_label = torch.cat([torch.ones(1), torch.zeros(len(clients) - 1)]).to(device)
                md_loss = contrastive(c_fake_emb, all_fake_emb, a_label, b_label)
                md_loss_list.append(md_loss.item())
                total_loss += md_loss
            elif args['L2G_use_logit_md_loss']:
                md_loss = torch.Tensor([0]).to(device)
                for s_id, student in enumerate(knowledge_exchanged_clients):
                    if s_id == t_id:
                        continue
                    _, s_fake_logit = student(fake_data, use_docking=False)
                    # md_loss += inverse_cross_entropy(s_fake_logit, F.softmax(c_fake_logit, dim=1).detach())
                    md_loss += inverse_cross_entropy(s_fake_logit, c_real_logit)
                md_loss = md_loss / (len(clients) - 1)
                md_loss_list.append(md_loss.item())
                total_loss += md_loss

            total_loss.backward()
            generator_optimizer.step()

        if args['log_wandb']:
            if emb_loss_list:
                wandb.log({'L2G G emb loss': np.mean(emb_loss_list)}, step=wandb_step + s)
            if cls_loss_list:
                wandb.log({'L2G G cls loss': np.mean(cls_loss_list)}, step=wandb_step + s)
            if md_loss_list:
                wandb.log({'L2G G md loss': np.mean(md_loss_list)}, step=wandb_step + s)

    return client_emb_logit_loaders


def generator_to_local(
        args,
        clients,
        client_optimizers,
        client_data_loaders,
        client_class_cnt,
        generator,
        pure_student,
        data_banks,
        client_emb_logit_loaders,
        G2L_epoch,
        batch_size,
        device='cpu',):
    num_clients, num_classes = client_class_cnt.shape

    ######################################## Generate Fake Data From Generator ########################################

    generator.eval()  # Enable BN to generate fake data
    for client in clients: client.eval()  # Freeze BN to distill original training knowledge

    # Generate Fake Data From Generator
    all_fake_data = []
    all_fake_target = []
    for c_id, client in enumerate(clients):
        fake_data = []
        fake_target = []
        for emb, logit, target in client_emb_logit_loaders[c_id]:
            emb, target = emb.to(device), target.to(device)
            y_one_hot = torch.zeros((len(target), num_classes)).to(device)
            y_one_hot.scatter_(1, target.unsqueeze(1), 1)
            c_fake_data = generator(emb, y_one_hot)
            fake_data.append(c_fake_data.detach())
            fake_target.append(target)
        fake_data = torch.cat(fake_data, dim=0).cpu()
        fake_target = torch.cat(fake_target, dim=0).cpu()
        all_fake_data.append(fake_data)
        all_fake_target.append(fake_target)

    # Normalize fake data
    combined_fake_data = torch.cat(all_fake_data, dim=0)
    mean = combined_fake_data.mean([0, 2, 3], keepdim=True)
    std = combined_fake_data.std([0, 2, 3], keepdim=True)
    print(f'>>Fake Data Mean: {mean.view(3)}, Std: {std.view(3)}')
    # if args['normalize_fake_data']:
    #     print(f'>> Normalizing Fake Data...')
    #     for i, fake_data in enumerate(all_fake_data):
    #         all_fake_data[i] = (fake_data - mean) / std

    # Prepare fake data loaders
    fake_datasets = []
    fake_data_loaders = []
    for c_id, client in enumerate(clients):
        fake_data = []
        fake_emb = []
        fake_logit = []
        fake_target = []
        temp_loader = DataLoader(CustomDataset(all_fake_data[c_id], all_fake_target[c_id]), batch_size=batch_size)
        with torch.no_grad():
            for c_fake_data, target in temp_loader:
                c_fake_data, target = c_fake_data.to(device), target.to(device)
                c_fake_emb, c_fake_logit = client(c_fake_data, use_docking=True)
                fake_data.append(c_fake_data.detach())
                fake_emb.append(c_fake_emb.detach())
                fake_logit.append(F.softmax(c_fake_logit, dim=1).detach())
                fake_target.append(target)
        fake_datasets.append(FakeDataset(torch.cat(fake_data, dim=0).cpu(), torch.cat(fake_emb, dim=0).cpu(), 
                                         torch.cat(fake_logit, dim=0).cpu(), torch.cat(fake_target, dim=0).cpu()))
        fake_data_loaders.append(DataLoader(fake_datasets[c_id], batch_size=batch_size, shuffle=True, drop_last=True))

    # Calculate G2L Iteration
    max_data_len = max([len(loader) for loader in fake_data_loaders])
    G2L_iteration = G2L_epoch * max_data_len

    data_bank_loaders = []
    # Preprocess data bank knowledge from each teacher (client)
    for c_id, client in enumerate(clients):
        if len(data_banks[c_id]) > 0:
            fake_data = []
            fake_emb = []
            fake_logit = []
            fake_target = []
            db_loader = DataLoader(data_banks[c_id], batch_size=batch_size, shuffle=True)
            with torch.no_grad():
                for data, target in islice(db_loader, G2L_iteration * args['G2L_data_bank_iteration']):
                    data, target = data.to(device), target.to(device)
                    c_fake_emb, c_fake_logit = client(data, use_docking=True)
                    fake_data.append(data.detach())
                    fake_emb.append(c_fake_emb.detach())
                    fake_logit.append(F.softmax(c_fake_logit, dim=1).detach())
                    fake_target.append(target)
            db_review_dataset = FakeDataset(torch.cat(fake_data, dim=0).cpu(), torch.cat(fake_emb, dim=0).cpu(),
                                            torch.cat(fake_logit, dim=0).cpu(), torch.cat(fake_target, dim=0).cpu())
            data_bank_loaders.append(DataLoader(db_review_dataset, batch_size=batch_size, shuffle=True, drop_last=True))

    #################################### Client Knowledge Exchange using Fake Data ####################################

    pure_student.train()
    pure_student_optimizer = optim.Adam(pure_student.parameters(), lr=args['client_lr'], weight_decay=args['reg'])
    for client in clients: client.train()
    client_inf_loaders = [InfiniteDataLoader(loader) for c_id, loader in enumerate(client_data_loaders)]
    fake_data_inf_loaders = [InfiniteDataLoader(loader) for c_id, loader in enumerate(fake_data_loaders)]
    data_bank_inf_loaders = [InfiniteDataLoader(loader) for c_id, loader in enumerate(data_bank_loaders)]

    # Convert Dict to List to save hashing time
    students = [[c_id, client, client_optimizers[c_id], client_inf_loaders[c_id]] for c_id, client in enumerate(clients)]
    wandb_step = wandb.run.step if args['log_wandb'] else 0

    for i in tqdm(range(G2L_iteration)):
        # Phase A
        real_loss_list = []
        real_acc_list = []
        # Phase B
        db_emb_loss_list = []
        db_logit_loss_list = []
        db_fake_acc_list = []
        # Extra
        db_ps_emb_loss_list = []
        db_ps_logit_loss_list = []
        # Phase C
        emb_loss_list = []
        logit_loss_list = []
        fake_acc_list = []
        # Extra
        ps_emb_loss_list = []
        ps_logit_loss_list = []

        # Phase A: Local Review Real Data
        for _ in range(args['G2L_local_review_iteration']):
            for c_id, client, client_optimizer, client_inf_loader in students:
                data, target = client_inf_loader.get_next()
                data, target = data.to(device), target.to(device)
                client_optimizer.zero_grad()
                _, c_real_logit = client(data)
                real_loss = F.cross_entropy(c_real_logit, target)
                real_loss.backward()
                client_optimizer.step()
                real_loss_list.append(real_loss.item())
                real_acc_list.append((torch.argmax(c_real_logit, dim=1) == target).float().mean().item())

        # Phase B: Review Data Bank Fake Data
        for _ in range(args['G2L_data_bank_iteration']):
            for data_bank_id, data_bank_inf_loader in enumerate(data_bank_inf_loaders):
                fake_data, t_fake_emb, t_fake_logit, target = data_bank_inf_loader.get_next()
                fake_data, t_fake_emb, t_fake_logit, target = fake_data.to(device), t_fake_emb.to(device), t_fake_logit.to(device), target.to(device)
                for c_id, client, client_optimizer, _ in students:
                    if c_id == data_bank_id:
                        continue
                    client_optimizer.zero_grad()
                    c_fake_emb, c_fake_logit = client(fake_data, use_docking=True)
                    loss = torch.Tensor([0]).to(device)
                    if args['G2L_cal_emb_loss']:
                        emb_loss = F.mse_loss(c_fake_emb, t_fake_emb)
                        db_emb_loss_list.append(emb_loss.item())
                        loss += emb_loss
                    logit_loss = F.kl_div(F.log_softmax(c_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
                    db_logit_loss_list.append(logit_loss.item())
                    loss += logit_loss
                    loss.backward()
                    client_optimizer.step()
                    db_fake_acc_list.append((torch.argmax(c_fake_logit, dim=1) == target).float().mean().item())

                # Extra: Train Pure Student to examine the performance of Knowledge Exchange
                pure_student_optimizer.zero_grad()
                ps_fake_emb, ps_fake_logit = pure_student(fake_data, use_docking=True)
                loss = torch.Tensor([0]).to(device)
                if args['G2L_cal_emb_loss']:
                    emb_loss = F.mse_loss(ps_fake_emb, t_fake_emb)
                    db_ps_emb_loss_list.append(emb_loss.item())
                    loss += emb_loss
                logit_loss = F.kl_div(F.log_softmax(ps_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
                db_ps_logit_loss_list.append(logit_loss.item())
                loss += logit_loss
                loss.backward()
                pure_student_optimizer.step()

        # Phase C: Train Current Fake Data
        for data_bank_id, data_bank_inf_loader in enumerate(fake_data_inf_loaders):
            fake_data, t_fake_emb, t_fake_logit, target = data_bank_inf_loader.get_next()
            fake_data, t_fake_emb, t_fake_logit, target = fake_data.to(device), t_fake_emb.to(device), t_fake_logit.to(device), target.to(device)
            for c_id, client, client_optimizer, _ in students:
                if c_id == data_bank_id:
                    continue
                client_optimizer.zero_grad()
                c_fake_emb, c_fake_logit = client(fake_data, use_docking=True)
                loss = torch.Tensor([0]).to(device)
                if args['G2L_cal_emb_loss']:
                    emb_loss = F.mse_loss(c_fake_emb, t_fake_emb)
                    emb_loss_list.append(emb_loss.item())
                    loss += emb_loss
                logit_loss = F.kl_div(F.log_softmax(c_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
                logit_loss_list.append(logit_loss.item())
                loss += logit_loss
                loss.backward()
                client_optimizer.step()
                fake_acc_list.append((torch.argmax(c_fake_logit, dim=1) == target).float().mean().item())

            # Extra: Train Pure Student to examine the performance of Knowledge Exchange
            pure_student_optimizer.zero_grad()
            ps_fake_emb, ps_fake_logit = pure_student(fake_data, use_docking=True)
            loss = torch.Tensor([0]).to(device)
            if args['G2L_cal_emb_loss']:
                emb_loss = F.mse_loss(ps_fake_emb, t_fake_emb)
                ps_emb_loss_list.append(emb_loss.item())
                loss += emb_loss
            logit_loss = F.kl_div(F.log_softmax(ps_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
            ps_logit_loss_list.append(logit_loss.item())
            loss += logit_loss
            loss.backward()
            pure_student_optimizer.step()

        if args['log_wandb']:
            if real_loss_list:
                wandb.log({'G2L local review Loss': np.mean(real_loss_list)}, step=wandb_step + i)
            if real_acc_list:
                wandb.log({'G2L local review acc': np.mean(real_acc_list)}, step=wandb_step + i)

            if db_emb_loss_list:
                wandb.log({'G2L db emb Loss': np.mean(db_emb_loss_list)}, step=wandb_step + i)
            if db_logit_loss_list:
                wandb.log({'G2L db cls Loss': np.mean(db_logit_loss_list)}, step=wandb_step + i)
            if db_fake_acc_list:
                wandb.log({'G2L db fake acc': np.mean(db_fake_acc_list)}, step=wandb_step + i)

            if db_ps_emb_loss_list:
                wandb.log({'G2L db Pure Student emb Loss': np.mean(db_ps_emb_loss_list)}, step=wandb_step + i)
            if db_ps_logit_loss_list:
                wandb.log({'G2L db Pure Student cls Loss': np.mean(db_ps_logit_loss_list)}, step=wandb_step + i)

            if emb_loss_list:
                wandb.log({'G2L emb Loss': np.mean(emb_loss_list)}, step=wandb_step + i)
            if logit_loss_list:
                wandb.log({'G2L cls Loss': np.mean(logit_loss_list)}, step=wandb_step + i)
            if fake_acc_list:
                wandb.log({'G2L fake acc': np.mean(fake_acc_list)}, step=wandb_step + i)

            if ps_emb_loss_list:
                wandb.log({'G2L Pure Student emb Loss': np.mean(ps_emb_loss_list)}, step=wandb_step + i)
            if ps_logit_loss_list:
                wandb.log({'G2L Pure Student cls Loss': np.mean(ps_logit_loss_list)}, step=wandb_step + i)

    return fake_datasets


def data_free_federated_knowledge_exchange(args, data_distributor):
    """
    Main function for Data-Free Federated Knowledge Exchange

    :param args: args dict
    :param data_distributor: DataDistributor object
    """
    device = args['device']
    n_class = data_distributor.n_class
    train_loaders = data_distributor.client_train_loaders
    fixed_train_loaders = data_distributor.client_fixed_train_loaders
    full_train_loader = data_distributor.full_train_loader
    full_test_loader = data_distributor.full_test_loader
    client_class_cnt = data_distributor.client_class_cnt

    print(">> Initializing clients models")
    clients = init_client_nets(args['n_clients'] + 1, args['client_encoder'], n_class, device)
    pure_student = clients[args['n_clients']]
    clients = clients[:args['n_clients']]
    client_optimizers = {}
    for client_id, client in enumerate(clients):
        client_optimizers[client_id] = optim.Adam(client.parameters(), lr=args['client_lr'], weight_decay=args['reg'])

    ############################################### Warmup Clients Model ###############################################

    mkdir(args['checkpoint_dir'])
    # Load the checkpoint if needed
    if args['load_clients'] is not None:
        file_name = get_checkpoint_file_name(args)
        checkpoint_path = args['load_clients'] + file_name + '.pt'
        # Check if the checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f'>> Checkpoint file {checkpoint_path} does not exist. Skip loading clients.')
            args['load_clients'] = None
        else:
            print(f'>> Loading clients checkpoint from {checkpoint_path}')
            state_dict = torch.load(checkpoint_path)
            for c_id, client in enumerate(clients):
                del state_dict[c_id]['model_state_dict']['docking.weight']
                del state_dict[c_id]['model_state_dict']['docking.bias']
                client.load_state_dict(state_dict[c_id]['model_state_dict'], strict=False)
                client_optimizers[c_id].load_state_dict(state_dict[c_id]['optimizer_state_dict'])
            if 'pure_student' in state_dict:
                pure_student.load_state_dict(state_dict['pure_student']['model_state_dict'], strict=False)
            print(f'>> Loaded.')

    # Client local training until converge
    if args['load_clients'] is None and args['warmup_clients']:
        # Initialize client optimizers
        print(">> Warmup Each Clients:")
        local_align_clients(
            clients=clients,
            optimizers=client_optimizers,
            train_loaders=train_loaders,
            passing_acc=args['local_align_acc'],
            test_loader=full_test_loader,
            log_wandb=args['log_wandb'],
            device=device)

        save_checkpoint(args, clients, client_optimizers, checkpoint_folder='warmup/')
        print(">> Warmup Clients Finished.")
    else:
        print(">> Skip Warmup Clients. If this is not intended, please modify configuration properly.")

    evaluate(clients, full_train_loader, 'Train', 'Local Aligned', 'cls_test', args['log_wandb'], device)
    evaluate(clients, full_test_loader, 'Test', 'Local Aligned', 'cls_test', args['log_wandb'], device)
    pure_student_evaluation(pure_student, full_train_loader, full_test_loader, args['log_wandb'], device)
    print("-------------------------------------------------------------------------------------------------")

    ################################################ Knowledge Exchange ################################################

    best_test_acc_mean = 0
    best_test_acc_std = 0
    data_banks = [CustomDataset([], []) for _ in range(args['n_clients'])]
    knowledge_exchanged_clients = [copy.deepcopy(client) for client in clients]

    for round_i in range(args['knowledge_exchange_rounds']):
        print(f'>> Current Round: {round_i}')

        if args['new_client_opt_every_round']:
            # Get a new client optimizer every round
            for client_id, client in enumerate(clients):
                client_optimizers[client_id] = \
                    optim.Adam(client.parameters(), lr=args['client_lr'], weight_decay=args['reg'])

        # Get a new generator every round
        generator = get_generator(model_name=args['generator_model'], nz=clients[0].output.in_features, n_cls=n_class)
        generator.to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args['generator_model_lr'])

        # L2G
        client_emb_logit_loaders = local_to_generator(
            args=args,
            clients=clients,
            client_data_loaders=fixed_train_loaders,
            client_aug_data_loaders=train_loaders,
            client_class_cnt=client_class_cnt,
            generator=generator,
            generator_optimizer=generator_optimizer,
            batch_size=args['batch_size'],
            L2G_epoch=args['L2G_epoch'],
            device=device,
            pure_student=pure_student,
            knowledge_exchanged_clients=knowledge_exchanged_clients,
        )
        # G2L
        fake_datasets = generator_to_local(
            args=args,
            clients=clients,
            client_optimizers=client_optimizers,
            client_data_loaders=train_loaders if args['G2L_augment_real'] else fixed_train_loaders,
            client_class_cnt=client_class_cnt,
            generator=generator,
            pure_student=pure_student,
            data_banks=data_banks,
            client_emb_logit_loaders=client_emb_logit_loaders,
            G2L_epoch=args['G2L_epoch'],
            batch_size=args['batch_size'],
            device=device,
        )
        # Finally, add the recently generated fake data into the data bank
        for c_id, data_bank in enumerate(data_banks):
            data_bank.update(fake_datasets[c_id].fake_data, fake_datasets[c_id].target)
        del fake_datasets
        # Save a copy of knowledge exchanged clients for next round adversarial loss calculation
        knowledge_exchanged_clients = [copy.deepcopy(client) for client in clients]
        # Evaluate after knowledge exchange
        pure_student.eval()
        pure_student_evaluation(pure_student, full_train_loader, full_test_loader, args['log_wandb'], device)
        for client in clients: client.eval()
        evaluate(clients, full_train_loader, 'Train', 'Global Exchanged', 'cls_test', args['log_wandb'], device)
        t_acc = evaluate(clients, full_test_loader, 'Test', 'Global Exchanged', 'cls_test', args['log_wandb'], device)
        if np.mean(t_acc) > best_test_acc_mean:
            best_test_acc_mean = np.mean(t_acc)
            best_test_acc_std = np.std(t_acc)
            save_checkpoint(args, clients, client_optimizers, checkpoint_folder='knowledge_exchange/')
        elif best_test_acc_mean - np.mean(t_acc) > 8:
            print("Early Stopping...")
            break

        if args['local_align_after_knowledge_exchange']:
            local_align_clients(
                clients=clients,
                optimizers=client_optimizers,
                train_loaders=train_loaders,
                passing_acc=args['local_align_acc'],
                test_loader=None,
                log_wandb=args['log_wandb'],
                device=device)

            # Evaluate after local alignment
            for client in clients: client.eval()
            evaluate(clients, full_train_loader, 'Train', 'Local Aligned', 'cls_test', args['log_wandb'], device)
            evaluate(clients, full_test_loader, 'Test', 'Local Aligned', 'cls_test', args['log_wandb'], device)

        if device == 'cuda':
            torch.cuda.empty_cache()
        print("-------------------------------------------------------------------------------------------------")

    print(f">> Best Test Accuracy after Knowledge Exchange: {best_test_acc_mean:.2f} ± {best_test_acc_std:.2f}")
    print("DFFKE Algorithm Ended.")