import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from DFFKE_losses import *
from dataset.utils_dataset import *

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




def data_free_federated_knowledge_exchange_with_latent_generator(
        args,
        clients,
        client_optimizers,
        client_data_loaders,
        client_aug_data_loaders,
        client_class_cnt,
        generator,
        generator_optimizer,
        batch_size,
        L2G_epoch,
        G2L_epoch,
        pure_student,
        data_banks,
        FKE_clients,
        device='cuda',):

    num_clients, num_classes = client_class_cnt.shape
    class_num = np.sum(client_class_cnt, axis=0)
    client_class_weight = client_class_cnt / (np.tile(class_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    class_client_weight = client_class_weight.transpose()
    L2G_client_data_loader = client_aug_data_loaders if args['L2G_augment_logits'] else client_data_loaders

    ###################################### Clustering Clients Data Distribution ######################################
    # Simulating the procedure of getting data embeddings from clients
    data_embeddings = defaultdict(lambda: defaultdict(list))
    for c_id, client in clients.items():
        client.eval()
        client_data_loader = L2G_client_data_loader[c_id]
        with torch.no_grad():
            for data, target in client_data_loader:
                data = data.to(device)
                embedding = client(data, use_docking=False)[0].detach()
                for i, y in enumerate(target):
                    data_embeddings[y.item()][c_id].append(embedding[i])

    client_data_embeddings = defaultdict(dict)
    for y in data_embeddings:
        for c_id in data_embeddings[y]:
            client_data_embeddings[y][c_id] = torch.stack(data_embeddings[y][c_id])
    del data_embeddings

    # Train the docking layer for each client to optimally cluster the data embeddings
    docking_params = []
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()
    for c_id, client in clients.items():
        docking_params.extend(client.docking.parameters())
    docking_optimizer = torch.optim.Adam(docking_params, lr=0.001)
    wandb_step = wandb.run.step if args['log_wandb'] else 0
    for s in tqdm(range(500)):
        class_embeddings = {}
        docking_optimizer.zero_grad()
        for y, embeddings in client_data_embeddings.items():
            class_embeddings[y] = torch.cat([clients[c_id].docking(embeddings[c_id]) for c_id in embeddings])
        class_means = {y: class_embeddings[y].mean(dim=0) for y in class_embeddings}

        class_means_tensor = []
        class_means_label = []
        for y, class_mean in class_means.items():
            class_means_tensor.append(class_mean)
            class_means_label.append(y)
        class_means_tensor = torch.stack(class_means_tensor)
        contrastive_loss = contrastive(class_means_tensor, class_means_tensor.detach())
        class_means = {y: class_means[y].detach().repeat((class_embeddings[y].shape[0], 1)) for y in class_means}
        mse_loss = torch.stack([MSELoss(class_embeddings[y], class_means[y]) for y in class_embeddings]).mean()
        l1_loss = torch.stack([L1Loss(class_embeddings[y], class_means[y]) for y in class_embeddings]).mean()
        (contrastive_loss + mse_loss).backward()
        docking_optimizer.step()
        if args['log_wandb']:
            wandb.log({'Clustering MSE Loss': mse_loss.item()}, step=wandb_step + s)
            wandb.log({'Clustering L1 Loss': l1_loss.item()}, step=wandb_step + s)
            wandb.log({'Clustering Contrastive Loss': contrastive_loss.item()}, step=wandb_step + s)

    # Collect the embeddings and logits of the clients
    client_emb_logit_loaders = {}
    for c_id, client in clients.items():
        client_emb = []
        client_logit = []
        client_target = []
        for data, target in L2G_client_data_loader[c_id]:
            data, target = data.to(device), target.to(device)
            emb, logit = client(data, use_docking=True)
            client_emb.append(emb.detach())
            client_logit.append(F.softmax(logit, dim=1).detach())
            client_target.append(target)
        client_emb = torch.cat(client_emb, dim=0).cpu()
        client_logit = torch.cat(client_logit, dim=0).cpu()
        client_target = torch.cat(client_target, dim=0).cpu()
        emb_logit_set = EmbLogitSet(client_emb, client_logit, client_target)
        client_emb_logit_loaders[c_id] = DataLoader(emb_logit_set, batch_size=batch_size, shuffle=True, drop_last=True)

    ############################################# L2G: Local to Generator ############################################

    wandb_step = wandb.run.step if args['log_wandb'] else 0
    generator.train()
    pure_student.eval()
    max_data_len = max([len(loader) for loader in client_emb_logit_loaders.values()])
    L2G_iteration = L2G_epoch * max_data_len
    for s in tqdm(range(L2G_iteration)):
        # Direct sample generator
        inf_emb_logit_loaders = {c_id: InfiniteDataLoader(loader) for c_id, loader in client_emb_logit_loaders.items()}
        emb_loss_list = []
        cls_loss_list = []
        md_loss_list = []
        for t_id, teacher in clients.items():
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
                for s_id, student in FKE_clients.items():
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
                for s_id, student in FKE_clients.items():
                    if s_id == t_id:
                        continue
                    _, s_fake_logit = student(fake_data, use_docking=False)
                    md_loss += inverse_cross_entropy(s_fake_logit, F.softmax(c_fake_logit, dim=1).detach())
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

    ############################################# G2L: Generator to Local ############################################
    # Generate fake data from generator, preprocess fake data knowledge from each teacher (client)
    fake_datasets = {}
    fake_data_loaders = {}
    data_bank_loaders = {c_id: None for c_id in clients}
    for c_id, client in clients.items():
        fake_data = []
        fake_emb = []
        fake_logit = []
        fake_target = []
        for emb, logit, target in client_emb_logit_loaders[c_id]:
            emb, target = emb.to(device), target.to(device)
            y_one_hot = torch.zeros((len(target), num_classes)).to(device)
            y_one_hot.scatter_(1, target.unsqueeze(1), 1)
            c_fake_data = generator(emb, y_one_hot)
            c_fake_emb, c_fake_logit = client(c_fake_data, use_docking=True)
            fake_data.append(c_fake_data.detach())
            fake_emb.append(c_fake_emb.detach())
            fake_logit.append(F.softmax(c_fake_logit, dim=1).detach())
            fake_target.append(target)

        fake_data = torch.cat(fake_data, dim=0).cpu()
        fake_emb = torch.cat(fake_emb, dim=0).cpu()
        fake_logit = torch.cat(fake_logit, dim=0).cpu()
        fake_target = torch.cat(fake_target, dim=0).cpu()
        fake_datasets[c_id] = FakeDataset(fake_data, fake_emb, fake_logit, fake_target)
        fake_data_loaders[c_id] = DataLoader(fake_datasets[c_id], batch_size=batch_size, shuffle=True, drop_last=True)

        # Preprocess data bank knowledge from each teacher (client)
        if data_banks[c_id]:
            fake_data = []
            fake_emb = []
            fake_logit = []
            fake_target = []
            prev_loader = DataLoader(data_banks[c_id], batch_size=batch_size, shuffle=True)
            for data, _, _, target in prev_loader:
                data, target = data.to(device), target.to(device)
                c_fake_emb, c_fake_logit = client(data, use_docking=True)
                fake_data.append(data.detach())
                fake_emb.append(c_fake_emb.detach())
                fake_logit.append(F.softmax(c_fake_logit, dim=1).detach())
                fake_target.append(target)
            data_banks[c_id].fake_data = np.array(torch.cat(fake_data, dim=0).cpu())
            data_banks[c_id].emb = np.array(torch.cat(fake_emb, dim=0).cpu())
            data_banks[c_id].logit = np.array(torch.cat(fake_logit, dim=0).cpu())
            data_banks[c_id].target = np.array(torch.cat(fake_target, dim=0).cpu())
            data_bank_loaders[c_id] = DataLoader(data_banks[c_id], batch_size=batch_size, shuffle=True, drop_last=True)


    federated_knowledge_exchange(
        args=args,
        clients=clients,
        client_optimizers=client_optimizers,
        client_data_loaders=client_aug_data_loaders if args['G2L_augment_real'] else client_data_loaders,
        client_class_cnt=client_class_cnt,
        proxy_data_loaders=fake_data_loaders,
        data_bank_loaders=data_bank_loaders,
        G2L_epoch=G2L_epoch,
        device=device,
        pure_student=pure_student,
    )

    for c_id, data_bank in data_banks.items():
        fake_dataset = fake_datasets[c_id]
        if data_bank:
            data_bank.update(fake_dataset)
        else:
            data_banks[c_id] = fake_dataset


def federated_knowledge_exchange(
        args,
        clients,
        client_optimizers,
        client_data_loaders,
        client_class_cnt,
        proxy_data_loaders,
        data_bank_loaders,
        G2L_epoch,
        pure_student,
        device='cpu',):
    pure_student.train()
    for c_id, client in clients.items():
        client.train()

    num_clients, num_classes = client_class_cnt.shape
    client_inf_loaders = {c_id: InfiniteDataLoader(loader) for c_id, loader in client_data_loaders.items()}
    fake_data_inf_loaders = {c_id: InfiniteDataLoader(loader) for c_id, loader in proxy_data_loaders.items()}
    data_bank_inf_loaders = {c_id: InfiniteDataLoader(loader) for c_id, loader in data_bank_loaders.items() if loader is not None}

    # Convert Dict to List to save hashing time
    students = [[c_id, client, client_optimizers[c_id], client_inf_loaders[c_id]] for c_id, client in clients.items()]

    import torch.optim as optimizer
    pure_student_optimizer = optimizer.Adam(pure_student.parameters(), lr=args['client_lr'], weight_decay=args['reg'])
    wandb_step = wandb.run.step if args['log_wandb'] else 0

    max_data_len = max([len(loader) for loader in proxy_data_loaders.values()])
    G2L_iteration = G2L_epoch * max_data_len

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
        for data_bank_id, data_bank_inf_loader in data_bank_inf_loaders.items():
            fake_data, t_fake_emb, t_fake_logit, target = data_bank_inf_loader.get_next()
            fake_data, t_fake_emb, t_fake_logit, target = fake_data.to(device), t_fake_emb.to(device), t_fake_logit.to(device), target.to(device)
            for c_id, client, client_optimizer, _ in students:
                if c_id == data_bank_id:
                    continue
                client_optimizer.zero_grad()
                c_fake_emb, c_fake_logit = client(fake_data, use_docking=args['dock_kd'])
                emb_loss = F.mse_loss(c_fake_emb, t_fake_emb)
                logit_loss = F.kl_div(F.log_softmax(c_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
                (emb_loss + logit_loss).backward()
                client_optimizer.step()
                db_emb_loss_list.append(emb_loss.item())
                db_logit_loss_list.append(logit_loss.item())
                db_fake_acc_list.append((torch.argmax(c_fake_logit, dim=1) == target).float().mean().item())

            # Extra: Train Pure Student to examine the performance of Knowledge Exchange
            pure_student_optimizer.zero_grad()
            ps_fake_emb, ps_fake_logit = pure_student(fake_data, use_docking=args['dock_kd'])
            emb_loss = F.mse_loss(ps_fake_emb, t_fake_emb)
            logit_loss = F.kl_div(F.log_softmax(ps_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
            (emb_loss + logit_loss).backward()
            pure_student_optimizer.step()
            db_ps_emb_loss_list.append(emb_loss.item())
            db_ps_logit_loss_list.append(logit_loss.item())

        # Phase C: Train Current Fake Data
        for data_bank_id, data_bank_inf_loader in fake_data_inf_loaders.items():
            fake_data, t_fake_emb, t_fake_logit, target = data_bank_inf_loader.get_next()
            fake_data, t_fake_emb, t_fake_logit, target = fake_data.to(device), t_fake_emb.to(device), t_fake_logit.to(device), target.to(device)
            for c_id, client, client_optimizer, _ in students:
                if c_id == data_bank_id:
                    continue
                client_optimizer.zero_grad()
                c_fake_emb, c_fake_logit = client(fake_data, use_docking=args['dock_kd'])
                emb_loss = F.mse_loss(c_fake_emb, t_fake_emb)
                logit_loss = F.kl_div(F.log_softmax(c_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
                (emb_loss + logit_loss).backward()
                client_optimizer.step()
                emb_loss_list.append(emb_loss.item())
                logit_loss_list.append(logit_loss.item())
                fake_acc_list.append((torch.argmax(c_fake_logit, dim=1) == target).float().mean().item())

            # Extra: Train Pure Student to examine the performance of Knowledge Exchange
            pure_student_optimizer.zero_grad()
            ps_fake_emb, ps_fake_logit = pure_student(fake_data, use_docking=args['dock_kd'])
            emb_loss = F.mse_loss(ps_fake_emb, t_fake_emb)
            logit_loss = F.kl_div(F.log_softmax(ps_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
            (emb_loss + logit_loss).backward()
            pure_student_optimizer.step()
            ps_emb_loss_list.append(emb_loss.item())
            ps_logit_loss_list.append(logit_loss.item())

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
