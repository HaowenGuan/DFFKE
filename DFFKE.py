import os
import wandb
import numpy as np
from tqdm import tqdm
from math import ceil
from itertools import islice
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.model_factory_fn import get_decoder, init_client_nets
from dataset.utils_dataset import EmbLogitSet, FakeDataset, InfiniteDataLoader, CustomDataset
from DFFKE_utils import mkdir, get_checkpoint_file_name, save_checkpoint, local_align_clients, evaluate, \
    pure_student_evaluation, save_decoder_visualization, approximate_model_sensitivity, estimate_file_size


def local_to_decoder(
        args,
        clients,
        client_data_loaders,
        client_class_cnt,
        decoder,
        decoder_optimizer,
        clustering_iteration,
        decoder_train_epoch,
        batch_size,
        additive_gaussian_DP=None,
        device='cuda', ):
    num_clients, num_classes = client_class_cnt.shape

    ###################################### Clustering Clients emb Distribution ######################################
    wandb_step = wandb.run.step if args['log_wandb'] else 0

    # Collect data embeddings from clients (In practice, clients upload embeddings to the server)
    client_emb_logit_loaders = {}
    n_samples = {}
    emb_cos_sim = []
    logit_cos_sim = []
    for c_id, client in clients:
        client.eval()  # Freeze BN to collect client data embeddings
        client_emb = []
        client_logit = []
        client_target = []
        with torch.no_grad():
            for data, target in client_data_loaders[c_id]:
                data, target = data.to(device), target.to(device)
                emb, logit = client(data, use_docking=False)
                ori_emb = emb.detach().clone()
                ori_logit = logit.detach().clone()
                # Additive Gaussian Differential Privacy
                if additive_gaussian_DP:
                    if "do_not_share_emb" in args and args["do_not_share_emb"]:
                        emb = torch.randn_like(emb)
                        logit = torch.randn_like(logit)
                    else:
                        emb += additive_gaussian_DP[0] * torch.randn_like(emb)
                        logit += additive_gaussian_DP[1] * torch.randn_like(logit)
                    emb_cos_sim.append(torch.nn.functional.cosine_similarity(ori_emb, emb).mean().item())
                    logit_cos_sim.append(torch.nn.functional.cosine_similarity(
                            F.softmax(ori_logit, dim=1),
                            F.softmax(logit, dim=1)
                        ).mean().item())
                client_emb.append(emb.detach())
                client_logit.append(F.softmax(logit, dim=1).detach())
                client_target.append(target)
        client_emb = torch.cat(client_emb, dim=0).cpu()
        client_logit = torch.cat(client_logit, dim=0).cpu()
        client_target = torch.cat(client_target, dim=0).cpu()
        emb_logit_set = EmbLogitSet(client_emb, client_logit, client_target)
        client_emb_logit_loaders[c_id] = DataLoader(emb_logit_set, batch_size=batch_size)
        n_samples[c_id] = len(client_target)
    if additive_gaussian_DP:
        print(f"Avg emb Cosine Similarity after apply Differential Privacy: {np.mean(emb_cos_sim):.4f}")
        print(f"Avg logit Cosine Similarity after apply Differential Privacy: {np.mean(logit_cos_sim):.4f}")

    # Train the docking layer for each client to optimally cluster the data embeddings into a unified embedding space
    cls_layer = nn.Linear(args['feature_dim'], args['num_classes']).to(device)
    clustering_params = list(cls_layer.parameters())
    for c_id, client in clients:
        clustering_params.extend(client.docking.parameters())
    docking_optimizer = torch.optim.Adam(clustering_params, lr=0.001, weight_decay=1e-3)
    if 'clustering_using_y' in args and args['clustering_using_y']:
        loss_function = lambda logit, y_hat, y: F.cross_entropy(logit, y, reduction='sum')
    else:
        loss_function = lambda logit, y_hat, y: F.kl_div(F.log_softmax(logit, dim=1), y_hat, reduction='sum')

    # balance_ratio = int(args['n_clients']) / int(args['participating_clients'])
    # clustering_iteration = int(clustering_iteration * balance_ratio)
    for s in tqdm(range(clustering_iteration)):
        clustering_loss = []
        for c_id, client in clients:
            emb_logit_loader = client_emb_logit_loaders[c_id]
            loss = torch.Tensor([0]).to(device)
            docking_optimizer.zero_grad()
            for emb, y_hat, target in emb_logit_loader:
                emb, y_hat, target = emb.to(device), y_hat.to(device), target.to(device)
                logit = cls_layer(client.docking(emb))
                loss += loss_function(logit, y_hat, target)
            loss /= n_samples[c_id]
            loss.backward()
            docking_optimizer.step()
            clustering_loss.append(loss.item())
        if args['log_wandb']:
            wandb.log({'Clustering Loss': np.mean(clustering_loss)}, step=wandb_step + s)

    # Collect the translated embeddings and logits of the clients
    client_emb_logit_loaders = {}
    for c_id, client in clients:
        client_emb = []
        client_logit = []
        client_target = []
        with torch.no_grad():
            for data, target in client_data_loaders[c_id]:
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

    ################################################ Training Decoder #################################################
    wandb_step = wandb.run.step if args['log_wandb'] else 0
    decoder.train()
    for c_id, client in clients: client.eval()  # Freeze BN to distill original training knowledge
    max_data_len = max([len(loader) for loader in client_emb_logit_loaders.values()])
    # balance_ratio = int(args['n_clients']) / int(args['participating_clients'])
    decoder_train_iteration = max(
        1000 // len(clients),
        int(decoder_train_epoch * max_data_len)
    )  # At least 1000 steps for decoder training
    fake_transform = args['resize_transform'] if 'resize_transform' in args else lambda x: x
    for s in tqdm(range(decoder_train_iteration)):
        inf_emb_logit_loaders = {c_id: InfiniteDataLoader(loader) for c_id, loader in client_emb_logit_loaders.items()}
        emb_loss_list = []
        for t_id, teacher in clients:
            c_real_emb, c_real_logit, target = inf_emb_logit_loaders[t_id].get_next()
            c_real_emb, c_real_logit, target = c_real_emb.to(device), c_real_logit.to(device), target.to(device)

            decoder_optimizer.zero_grad()
            # y_one_hot = torch.zeros((len(target), num_classes)).to(device)
            # y_one_hot.scatter_(1, target.unsqueeze(1), 1)
            # If using ClassLatentGenerator, pick y_one_hot or c_real_logit as the condition to decoder
            fake_data = decoder(c_real_emb, c_real_logit)
            fake_data = fake_transform(fake_data)
            c_fake_emb, c_fake_logit = teacher(fake_data, use_docking=True)
            total_loss = torch.Tensor([0]).to(device)

            # Data Fidelity Loss
            fid_loss = F.mse_loss(c_fake_emb, c_real_emb)
            emb_loss_list.append(fid_loss.item())
            total_loss += fid_loss

            # if args['decoder_model'] == 'LatentGenerator':
            #     # Part of Fidelity Loss, if decoder is ClassLatentGenerator
            #     cls_loss = F.kl_div(F.log_softmax(c_fake_logit, dim=1), c_real_logit, reduction='batchmean')
            #     total_loss += cls_loss

            total_loss.backward()
            decoder_optimizer.step()

        if args['log_wandb']:
            if emb_loss_list:
                wandb.log({'Decoder emb loss': np.mean(emb_loss_list)}, step=wandb_step + s)

    return client_emb_logit_loaders


def decoder_to_local(
        args,
        clients,
        client_optimizers,
        client_class_cnt,
        decoder,
        pure_student,
        memory_buffers,
        client_emb_logit_loaders,
        FKE_epoch,
        batch_size,
        device='cpu', ):
    num_clients, num_classes = client_class_cnt.shape
    FKE_communication_cost = 0

    ###################################### Generate Synthetic Data From Decoder ########################################
    decoder.eval()  # Enable BN to generate fake data
    for c_id, client in clients:
        client.eval()  # Freeze BN to distill original training knowledge

    # Prepare fake data loaders
    fake_data_loaders = {}
    fake_transform = args['resize_transform'] if 'resize_transform' in args else lambda x: x
    for c_id, client in clients:
        fake_data = []
        fake_target = []
        for emb, logit, target in client_emb_logit_loaders[c_id]:
            emb, logit, target = emb.to(device), logit.to(device), target.to(device)
            # y_one_hot = torch.zeros((len(target), num_classes)).to(device)
            # y_one_hot.scatter_(1, target.unsqueeze(1), 1)
            # If using ClassLatentGenerator, choose y_one_hot or logit to use as condition for decoder
            c_fake_data = decoder(emb, logit)
            fake_data.append(c_fake_data.detach().cpu())
            fake_target.append(target.cpu())
        fake_data = torch.cat(fake_data, dim=0)
        fake_target = torch.cat(fake_target, dim=0)

        # Check fake data normalization, mean should be close to 0 and std should be close to 1
        if c_id == 0:
            mean = fake_data.mean([0, 2, 3], keepdim=True)
            std = fake_data.std([0, 2, 3], keepdim=True)
            print(f'>>Synthetic Data Mean: {mean.view(3)}, Std: {std.view(3)}')

        temp_loader = DataLoader(CustomDataset(fake_data, fake_target), batch_size=batch_size)
        fake_emb = []
        fake_logit = []
        with torch.no_grad():
            for c_fake_data, target in temp_loader:
                c_fake_data, target = c_fake_data.to(device), target.to(device)
                c_fake_data = fake_transform(c_fake_data)
                c_fake_emb, c_fake_logit = client(c_fake_data, use_docking=True)
                fake_emb.append(c_fake_emb.detach().cpu())
                fake_logit.append(F.softmax(c_fake_logit, dim=1).detach().cpu())
        # Each clients share its logits to n - 1 clients
        FKE_communication_cost += estimate_file_size(torch.cat(fake_logit, dim=0)) * (len(clients) - 1)
        fake_dataset = FakeDataset(fake_data, torch.cat(fake_emb, dim=0), torch.cat(fake_logit, dim=0), fake_target)
        fake_data_loaders[c_id] = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Calculate FKE Iteration
    max_data_len = max([len(loader) for loader in fake_data_loaders.values()])
    # balance_ratio = int(args['n_clients']) / int(args['participating_clients'])
    minimum_FKE_iteration = int(1000 / (1 + args['memory_buffer_iteration']) / (len(clients) - 1))
    FKE_iteration = max(minimum_FKE_iteration, int(FKE_epoch * max_data_len))

    # Preprocess memory buffer knowledge from each teacher (client)
    memory_buffer_iteration = FKE_iteration * args['memory_buffer_iteration']
    memory_buffer_loaders = {}
    for c_id, client in clients:
        if len(memory_buffers[c_id]) > 0 and memory_buffer_iteration > 0:
            # Each client will update `memory_buffer_iteration` times on the memory buffer.
            # The memory buffer may store more than one past round of emb-logit pairs.
            # So we evenly sample data from each past round to review in the current round
            n_batch_per_past_round = ceil(memory_buffer_iteration / len(memory_buffers[c_id]))
            fake_data = []
            fake_target = []
            with torch.no_grad():
                for buffered_decoder, emb_logit_loader in memory_buffers[c_id]:
                    buffered_decoder.to(device)
                    buffered_decoder.eval()
                    for emb, logit, target in islice(emb_logit_loader, n_batch_per_past_round):
                        emb, logit, target = emb.to(device), logit.to(device), target.to(device)
                        # y_one_hot = torch.zeros((len(target), num_classes)).to(device)
                        # y_one_hot.scatter_(1, target.unsqueeze(1), 1)
                        # If using ClassLatentGenerator, choose y_one_hot or logit to use as condition for decoder
                        c_fake_data = buffered_decoder(emb, logit)
                        fake_data.append(c_fake_data.detach().cpu())
                        fake_target.append(target.cpu())
                    buffered_decoder.to('cpu')
            fake_data = torch.cat(fake_data, dim=0).cpu()
            fake_target = torch.cat(fake_target, dim=0).cpu()
            memory_loader = DataLoader(CustomDataset(fake_data, fake_target), batch_size=batch_size)
            fake_emb = []
            fake_logit = []
            with torch.no_grad():
                for c_fake_data, target in memory_loader:
                    c_fake_data, target = c_fake_data.to(device), target.to(device)
                    c_fake_data = fake_transform(c_fake_data)
                    c_fake_emb, c_fake_logit = client(c_fake_data, use_docking=True)
                    fake_emb.append(c_fake_emb.detach().cpu())
                    fake_logit.append(F.softmax(c_fake_logit, dim=1).detach().cpu())
            memory_review_dataset = FakeDataset(fake_data, torch.cat(fake_emb, dim=0),
                                                torch.cat(fake_logit, dim=0), fake_target)
            # Each clients share its logits to n - 1 clients
            FKE_communication_cost += estimate_file_size(torch.cat(fake_logit, dim=0)) * (len(clients) - 1)
            memory_buffer_loaders[c_id] = DataLoader(
                memory_review_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )

    ################################ Federated Knowledge Exchange using Synthetic Data ################################
    pure_student.train()
    pure_student_optimizer = optim.Adam(pure_student.parameters(), lr=args['client_lr'], weight_decay=args['reg'])
    for c_id, client in clients:
        client.train()
    fake_data_inf_loaders = [[c_id, InfiniteDataLoader(loader)] for c_id, loader in fake_data_loaders.items()]
    memory_buffer_inf_loaders = [[c_id, InfiniteDataLoader(loader)] for c_id, loader in memory_buffer_loaders.items()]

    # Convert Dict to List to save hashing time
    students = [[c_id, client, client_optimizers[c_id]] for c_id, client in clients]
    wandb_step = wandb.run.step if args['log_wandb'] else 0

    for i in tqdm(range(FKE_iteration)):
        # Review memory buffer
        mem_emb_loss_list = []
        mem_logit_loss_list = []
        mem_fake_acc_list = []
        # Extra
        mem_ps_emb_loss_list = []
        mem_ps_logit_loss_list = []
        # Federated Knowledge Exchange
        emb_loss_list = []
        logit_loss_list = []
        fake_acc_list = []
        # Extra
        ps_emb_loss_list = []
        ps_logit_loss_list = []

        # Review synthetic data in memory buffer
        for _ in range(args['memory_buffer_iteration']):
            for memory_client_id, memory_buffer_inf_loader in memory_buffer_inf_loaders:
                fake_data, t_fake_emb, t_fake_logit, target = memory_buffer_inf_loader.get_next()
                fake_data, t_fake_emb, t_fake_logit, target = (fake_data.to(device), t_fake_emb.to(device),
                                                               t_fake_logit.to(device), target.to(device))
                fake_data = fake_transform(fake_data)
                for c_id, client, client_optimizer in students:
                    if c_id == memory_client_id:
                        continue
                    client_optimizer.zero_grad()
                    c_fake_emb, c_fake_logit = client(fake_data, use_docking=True)
                    loss = torch.Tensor([0]).to(device)

                    emb_loss = F.mse_loss(c_fake_emb, t_fake_emb)
                    mem_emb_loss_list.append(emb_loss.item())
                    loss += emb_loss

                    logit_loss = F.kl_div(F.log_softmax(c_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
                    mem_logit_loss_list.append(logit_loss.item())
                    loss += logit_loss

                    loss.backward()
                    client_optimizer.step()
                    mem_fake_acc_list.append((torch.argmax(c_fake_logit, dim=1) == target).float().mean().item())

                # Extra: Train a Pure Student, who only learns, to examine the performance of Knowledge Exchange
                pure_student_optimizer.zero_grad()
                ps_fake_emb, ps_fake_logit = pure_student(fake_data, use_docking=True)
                loss = torch.Tensor([0]).to(device)

                emb_loss = F.mse_loss(ps_fake_emb, t_fake_emb)
                mem_ps_emb_loss_list.append(emb_loss.item())
                loss += emb_loss

                logit_loss = F.kl_div(F.log_softmax(ps_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
                mem_ps_logit_loss_list.append(logit_loss.item())
                loss += logit_loss

                loss.backward()
                pure_student_optimizer.step()

        # Federated Knowledge Exchange
        for teacher_id, fake_data_inf_loader in fake_data_inf_loaders:
            fake_data, t_fake_emb, t_fake_logit, target = fake_data_inf_loader.get_next()
            fake_data, t_fake_emb, t_fake_logit, target = (fake_data.to(device), t_fake_emb.to(device),
                                                           t_fake_logit.to(device), target.to(device))
            fake_data = fake_transform(fake_data)
            for c_id, client, client_optimizer in students:
                if c_id == teacher_id:
                    continue
                client_optimizer.zero_grad()
                c_fake_emb, c_fake_logit = client(fake_data, use_docking=True)
                loss = torch.Tensor([0]).to(device)

                emb_loss = F.mse_loss(c_fake_emb, t_fake_emb)
                emb_loss_list.append(emb_loss.item())
                loss += emb_loss

                logit_loss = F.kl_div(F.log_softmax(c_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
                logit_loss_list.append(logit_loss.item())
                loss += logit_loss

                loss.backward()
                client_optimizer.step()
                fake_acc_list.append((torch.argmax(c_fake_logit, dim=1) == target).float().mean().item())

            # Extra: Train a Pure Student, who only learns, to examine the performance of Knowledge Exchange
            pure_student_optimizer.zero_grad()
            ps_fake_emb, ps_fake_logit = pure_student(fake_data, use_docking=True)
            loss = torch.Tensor([0]).to(device)

            emb_loss = F.mse_loss(ps_fake_emb, t_fake_emb)
            ps_emb_loss_list.append(emb_loss.item())
            loss += emb_loss

            logit_loss = F.kl_div(F.log_softmax(ps_fake_logit, dim=1), t_fake_logit, reduction='batchmean')
            ps_logit_loss_list.append(logit_loss.item())
            loss += logit_loss

            loss.backward()
            pure_student_optimizer.step()

        if args['log_wandb']:
            if mem_emb_loss_list:
                wandb.log({'FKE mem emb Loss': np.mean(mem_emb_loss_list)}, step=wandb_step + i)
            if mem_logit_loss_list:
                wandb.log({'FKE mem cls Loss': np.mean(mem_logit_loss_list)}, step=wandb_step + i)
            if mem_fake_acc_list:
                wandb.log({'FKE mem fake acc': np.mean(mem_fake_acc_list)}, step=wandb_step + i)

            if mem_ps_emb_loss_list:
                wandb.log({'FKE mem Pure Student emb Loss': np.mean(mem_ps_emb_loss_list)}, step=wandb_step + i)
            if mem_ps_logit_loss_list:
                wandb.log({'FKE mem Pure Student cls Loss': np.mean(mem_ps_logit_loss_list)}, step=wandb_step + i)

            if emb_loss_list:
                wandb.log({'FKE emb Loss': np.mean(emb_loss_list)}, step=wandb_step + i)
            if logit_loss_list:
                wandb.log({'FKE cls Loss': np.mean(logit_loss_list)}, step=wandb_step + i)
            if fake_acc_list:
                wandb.log({'FKE fake acc': np.mean(fake_acc_list)}, step=wandb_step + i)

            if ps_emb_loss_list:
                wandb.log({'FKE Pure Student emb Loss': np.mean(ps_emb_loss_list)}, step=wandb_step + i)
            if ps_logit_loss_list:
                wandb.log({'FKE Pure Student cls Loss': np.mean(ps_logit_loss_list)}, step=wandb_step + i)

    print(f'FKE Communication Cost: {FKE_communication_cost:.2f} MB')
    return FKE_communication_cost


def data_free_federated_knowledge_exchange(args, data_distributor):
    """
    Main function for Data-Free Federated Knowledge Exchange

    :param args: args dict
    :param data_distributor: DataDistributor object
    """
    device = args['device']
    n_clients = args['n_clients']
    n_class = data_distributor.n_class
    img_size = data_distributor.img_size
    train_loaders = data_distributor.client_train_loaders
    non_aug_train_loaders = data_distributor.non_aug_train_loaders
    full_train_loader = data_distributor.full_train_loader
    full_test_loader = data_distributor.full_test_loader
    client_test_loaders = data_distributor.client_test_loaders
    client_class_cnt = data_distributor.client_class_cnt
    client_visual_loaders = data_distributor.client_visual_loaders
    eval_gap = int(1 / args['join_ratio'])
    n_participants = int(n_clients * args['join_ratio'])
    print(f'>> evaluation gap: {eval_gap}')
    print(f'>> Number of Participants each round: {n_participants}')

    if args['resize_img'] % img_size != 0:
        raise ValueError(f"resize_img ({args['resize_img']}) need be a multiple of img_size ({img_size})")
    args['resize_transform'] = nn.Upsample(scale_factor=args['resize_img'] // img_size)
    print(f'>> Original Image Size: {img_size}, resizing: {args["resize_img"]}')
    print(f'>> Resize Transform: {args["resize_transform"]}')

    print(">> Initializing clients models")
    clients = init_client_nets(
        num_clients=args['n_clients'],
        model_choice=args['models'],
        n_class=n_class,
        docking_dim=args['feature_dim'],
        get_pure_student=True,
        device=device
    )
    pure_student = clients.pop()
    clients = [[c_id, client] for c_id, client in enumerate(clients)]
    if 'participating_clients' in args:
        # Ablation control to limit the number of participating clients
        clients = clients[:args['participating_clients']]
    client_optimizers = []
    for client_id, client in clients:
        client_optimizers.append(optim.Adam(client.parameters(), lr=args['client_lr'], weight_decay=args['reg']))

    ############################################### Warmup Clients Model ###############################################
    mkdir(args['checkpoint_dir'])
    # Load the checkpoint if needed
    if args['load_clients'] is not None:
        file_name = get_checkpoint_file_name(args)
        checkpoint_path = str(os.path.join(args['load_clients'], file_name))
        # Check if the checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f'>> Checkpoint file {checkpoint_path} does not exist. Skip loading clients and proceed to Warmup.')
            args['load_clients'] = None
        else:
            print(f'>> Loading checkpoint from {checkpoint_path}')
            state_dict = torch.load(checkpoint_path)
            for c_id, client in clients:
                del state_dict[c_id]['model_state_dict']['docking.weight']
                del state_dict[c_id]['model_state_dict']['docking.bias']
                client.load_state_dict(state_dict[c_id]['model_state_dict'], strict=False)
                # client_optimizers[c_id].load_state_dict(state_dict[c_id]['optimizer_state_dict'])
            print(f'>> Clients checkpoint Loaded.')
            if 'pure_student' in state_dict:
                pure_student.load_state_dict(state_dict['pure_student']['model_state_dict'], strict=False)
                print(f'>> Pure Student checkpoint Loaded.')

    if args['load_clients'] is None:
        # Client local training private until reaching target accuracy on train set
        print(">> Warming-up Clients:")
        local_align_clients(
            clients=clients,
            optimizers=client_optimizers,
            train_loaders=train_loaders,
            passing_acc=args['local_align_acc'],
            test_loaders=client_test_loaders if len(client_test_loaders) > 0 else None,
            log_wandb=args['log_wandb'],
            device=device
        )
        print(">> Warmup Clients Finished.")
        if args['save_clients']:
            save_checkpoint(args, clients, client_optimizers, checkpoint_folder='warmup/', pure_student=pure_student)
    else:
        print(">> Loaded Pretrained Clients. Skip Warmup Clients.")

    additive_gaussian_DP = None
    if 'additive_gaussian_DP' in args and args['additive_gaussian_DP']:
        emd_sensitivity, logit_sensitivity = approximate_model_sensitivity(clients, non_aug_train_loaders, device)
        print(f'>> Model emb Sensitivity: {emd_sensitivity:.2f}, Model logit Sensitivity: {logit_sensitivity:.2f}')
        one_over_delta = np.mean(np.sum(client_class_cnt, axis=1))  # 1 / delta = dataset size
        epsilon = float(args['additive_gaussian_DP'])  # Privacy Budget
        emd_std = np.sqrt(2 * np.log(1.25 * one_over_delta)) * emd_sensitivity / epsilon
        logit_std = np.sqrt(2 * np.log(1.25 * one_over_delta)) * logit_sensitivity / epsilon
        additive_gaussian_DP = [emd_std, logit_std]
        print(f'>> Additive Gaussian Noise std emd/logits: {additive_gaussian_DP[0]:.2f}/{additive_gaussian_DP[1]:.2f}')

    # evaluate(clients, [full_train_loader] * n_clients, 'Train', 'Local Aligned', args['log_wandb'], device)
    evaluate(clients, [full_test_loader] * n_clients, 'Test', 'Local Aligned', args['log_wandb'], device)
    evaluate(clients, client_test_loaders, 'Personalized Test', 'Local Aligned', args['log_wandb'], device)
    pure_student_evaluation(pure_student, full_train_loader, full_test_loader, args['log_wandb'], device)
    print("-------------------------------------------------------------------------------------------------")

    ###################################################### DFFKE ######################################################
    best_full_test_acc_mean = 0
    best_full_test_acc_std = 0
    best_personalized_test_acc_mean = 0
    best_personalized_test_acc_std = 0
    memory_buffers = [deque() for _ in range(n_clients)]
    communication_cost_list = []

    for round_i in range(1, args['knowledge_exchange_rounds'] + 1):
        print(f'>> Current Round: {round_i}')
        selected_clients = set(np.random.choice(n_clients, n_participants, replace=False))
        print(f'>> Selected Clients: {selected_clients}')
        selected_clients = [[c_id, client] for c_id, client in clients if c_id in selected_clients]

        # Initialize client optimizer
        for client_id, client in selected_clients:
            client_optimizers[client_id] = \
                optim.Adam(client.parameters(), lr=args['client_lr'], weight_decay=args['reg'])

        # Initialize decoder
        decoder = get_decoder(
            model_name=args['decoder_model'],
            nz=args['feature_dim'],
            n_cls=n_class,
            img_size=img_size
        )
        decoder.to(device)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args['decoder_model_lr'])

        # Decoder Training: Distributed Embedding Alignment + Embedding Guided Decoder Training
        client_emb_logit_loaders = local_to_decoder(
            args=args,
            clients=selected_clients,
            client_data_loaders=non_aug_train_loaders,
            client_class_cnt=client_class_cnt,
            decoder=decoder,
            decoder_optimizer=decoder_optimizer,
            clustering_iteration=args['clustering_iteration'],
            decoder_train_epoch=args['decoder_train_epoch'],
            batch_size=args['batch_size'],
            additive_gaussian_DP=additive_gaussian_DP,
            device=device,
        )

        # Calculate data-free module communication cost
        communication_cost = 0
        for c_id, client in selected_clients:
            communication_cost += estimate_file_size(client.state_dict())
        print(f'>> Communication Cost for Uploading Clients models: {communication_cost:.2f} MB')
        temp = 0
        for emb_loader in client_emb_logit_loaders.values():
            temp += estimate_file_size(emb_loader.dataset.emb) * 2
            communication_cost += estimate_file_size(emb_loader.dataset.emb) * 2  # two times, (1) to (2), (3) to (4)
        print(f'>> Communication Cost for Uploading Clients Embeddings and Logits: {temp:.2f} MB')
        communication_cost += estimate_file_size(decoder.state_dict())
        print(f'>> Communication Cost for Uploading Decoder: {estimate_file_size(decoder.state_dict()):.2f} MB')

        # Visualize decoder output when needed
        if 'visualize_decoder_output' in args and args['visualize_decoder_output']:
            save_decoder_visualization(args['checkpoint_dir'], round_i, clients, decoder,
                                       client_visual_loaders, n_class, device)

        # FKE: Collect Synthetic Data + Federated Knowledge Exchange
        communication_cost += decoder_to_local(
            args=args,
            clients=selected_clients,
            client_optimizers=client_optimizers,
            client_class_cnt=client_class_cnt,
            decoder=decoder,
            pure_student=pure_student,
            memory_buffers=memory_buffers,
            client_emb_logit_loaders=client_emb_logit_loaders,
            FKE_epoch=args['FKE_epoch'],
            batch_size=args['batch_size'],
            device=device,
        )

        # Lastly, store the decoder and embeddings into the memory buffer
        decoder.to('cpu')  # Save the buffered decoder to CPU, activate it to GPU when used again
        for c_id, _ in selected_clients:
            client_memory_buffer = memory_buffers[c_id]
            client_memory_buffer.append([decoder, client_emb_logit_loaders[c_id]])
            # Ablation control for memory buffer limit
            if 'memory_limit' in args and args['memory_limit'] and len(client_memory_buffer) > args['memory_limit']:
                client_memory_buffer.popleft()
                print(f'>> Memory Buffer of Client {c_id} is full. Limit: {args["memory_limit"]}. Dump the oldest data.')
            if c_id == 0:
                client_memory_buffer_size = 0
                for buffered_decoder, emb_loader in client_memory_buffer:
                    client_memory_buffer_size += estimate_file_size(buffered_decoder.state_dict())
                    client_memory_buffer_size += estimate_file_size(emb_loader.dataset.emb)
                print(f'>> Current Memory Buffer Size of Client 0: {client_memory_buffer_size:.2f} MB')

        if round_i % eval_gap == 0:
            # Evaluate after knowledge exchange
            pure_student_evaluation(pure_student, full_train_loader, full_test_loader, args['log_wandb'], device)
            # evaluate(clients, [full_train_loader] * n_clients, 'Train', 'Global Exchanged', args['log_wandb'], device)
            ft_acc = evaluate(clients, [full_test_loader] * n_clients,
                              'Test', 'Global Exchanged', args['log_wandb'], device)
            evaluate(clients, client_test_loaders,
                     'Personalized Test', 'Global Exchanged', args['log_wandb'], device)

            # Save the best model
            if np.mean(ft_acc) > best_full_test_acc_mean:
                best_full_test_acc_mean = np.mean(ft_acc)
                best_full_test_acc_std = np.std(ft_acc)
                # if args['save_clients']:
                #     save_checkpoint(args, clients, client_optimizers, checkpoint_folder='knowledge_exchange/',
                #                     pure_student=pure_student)
            elif (best_full_test_acc_mean - np.mean(ft_acc)) / best_full_test_acc_mean > 0.1:
                # If the test accuracy drops more than 10% from peak, stop DFFKE
                print("Early Stopping DFFKE ...")
                break

        # Local Training Private Dataset
        local_align_clients(
            clients=selected_clients,
            optimizers=client_optimizers,
            train_loaders=train_loaders,
            passing_acc=args['local_align_acc'],
            test_loaders=None,
            log_wandb=args['log_wandb'],
            device=device
        )

        if round_i % eval_gap == 0:
            # Evaluate after local alignment
            # evaluate(clients, [full_train_loader] * n_clients, 'Train', 'Local Aligned', args['log_wandb'], device)
            evaluate(clients, [full_test_loader] * n_clients,
                     'Test', 'Local Aligned', args['log_wandb'], device)
            pt_acc = evaluate(clients, client_test_loaders,
                              'Personalized Test', 'Local Aligned', args['log_wandb'], device)

            if np.mean(pt_acc) > best_personalized_test_acc_mean:
                best_personalized_test_acc_mean = np.mean(pt_acc)
                best_personalized_test_acc_std = np.std(pt_acc)

        print(f'>> Communication Cost: {communication_cost:.2f} MB')
        communication_cost_list.append(communication_cost)
        if device == 'cuda':
            torch.cuda.empty_cache()
        print("-------------------------------------------------------------------------------------------------")

    print(f">> Best Test Accuracy after Knowledge Exchange:",
          f"{best_full_test_acc_mean:.2f}$\pm${best_full_test_acc_std:.2f}")
    print(f">> Best Personalized Test Accuracy after Knowledge Exchange:",
          f"{best_personalized_test_acc_mean:.2f}$\pm${best_personalized_test_acc_std:.2f}")
    print(f">> Total Communication Cost: {sum(communication_cost_list) / 1024:.2f} GB")
    print("DFFKE Algorithm Ended.")
