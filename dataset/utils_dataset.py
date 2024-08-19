from collections import defaultdict

import torch.utils.data as data
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import os
import torch


class CustomDataset(data.Dataset):
    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.data = np.array(data)
        self.target = np.array(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class EmbLogitSet(data.Dataset):
    def __init__(self, emb, logit, target):
        self.emb = np.array(emb)
        self.logit = np.array(logit)
        self.target = np.array(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (emb, logit, target) where target is index of the target class.
        """
        emb, logit, target = self.emb[index], self.logit[index], self.target[index]
        return emb, logit, target

    def __len__(self):
        return len(self.emb)


class FakeDataset(data.Dataset):
    def __init__(self, fake_data, emb, logit, target):
        self.fake_data = np.array(fake_data)
        self.emb = np.array(emb)
        self.logit = np.array(logit)
        self.target = np.array(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (fake_data, emb, logit) where target is index of the target class.
        """
        fake_data, emb, logit, target = self.fake_data[index], self.emb[index], self.logit[index], self.target[index]
        return fake_data, emb, logit, target

    def __len__(self):
        return len(self.fake_data)

    def update(self, other_dataset: 'FakeDataset'):
        self.fake_data = np.concatenate((self.fake_data, other_dataset.fake_data))
        self.emb = np.concatenate((self.emb, other_dataset.emb))
        self.logit = np.concatenate((self.logit, other_dataset.logit))
        self.target = np.concatenate((self.target, other_dataset.target))


class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def get_next(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)


def load_cifar10_data(data_dir):
    print(f"Loading CIFAR10 data...")
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train = CIFAR10(data_dir, train=True, transform=transform, download=True)
    cifar10_test = CIFAR10(data_dir, train=False, transform=transform, download=True)

    X_train, Y_train = cifar10_train.data, np.array(cifar10_train.targets)
    X_test, Y_test = cifar10_test.data, np.array(cifar10_test.targets)
    n_class = len(cifar10_train.classes)

    return X_train, Y_train, X_test, Y_test, n_class


def load_cifar100_data(data_dir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train = CIFAR100(data_dir, train=True, transform=transform, download=True)
    cifar100_test = CIFAR100(data_dir, train=False, transform=transform, download=True)

    X_train, Y_train = cifar100_train.data, np.array(cifar100_train.targets)
    X_test, Y_test = cifar100_test.data, np.array(cifar100_test.targets)
    n_class = len(cifar100_train.classes)

    return X_train, Y_train, X_test, Y_test, n_class


def load_fc100_data(data_dir):
    X_train, Y_train, X_test, Y_test, n_class = load_cifar100_data(data_dir)

    X_total = np.concatenate([X_train, X_test], 0)
    y_total = np.concatenate([Y_train, Y_test], 0)

    test_data_idxs = []
    for k in fine_split['test']:
        test_data_idxs.extend(np.where(y_total == k)[0].tolist())

    X_test = X_total[test_data_idxs]
    Y_test = y_total[test_data_idxs]

    train_data_idx = []
    for k in fine_split['train']:
        train_data_idx.extend(np.where(y_total == k)[0].tolist())

    X_train = X_total[train_data_idx]
    Y_train = y_total[train_data_idx]

    return X_train, Y_train, X_test, Y_test, n_class


def load_dataset(dataset, data_dir):
    if dataset == 'CIFAR10':
        print(f"Loading CIFAR10 data...")
        return load_cifar10_data(data_dir)
    elif dataset == 'CIFAR100':
        print(f"Loading CIFAR100 data...")
        return load_cifar100_data(data_dir)
    elif dataset == 'FC100':
        print(f"Loading FC100 data...")
        return load_fc100_data(data_dir)
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")


def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in range(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data


def get_federated_learning_dataset(dataset, data_dir, n_clients, alpha=1.0, sampling_ratio=1.0, redo_split=False):
    buffer_path = data_dir + f'buffer/{dataset}_{n_clients}client_{float(alpha)}alpha-{float(sampling_ratio)}ratio.pt'

    if os.path.exists(buffer_path) and not redo_split:
        print(f'>> Loading buffer dataset from {buffer_path}')
        buffer = torch.load(buffer_path)
        X_train_clients = np.array((buffer['X_train_clients'])).copy()
        Y_train_clients = np.array(buffer['Y_train_clients']).copy()
        X_test = np.array(buffer['X_test']).copy()
        Y_test = np.array(buffer['Y_test']).copy()
        client_class_cnt = np.array(buffer['client_class_cnt']).copy()
        del buffer
        print(f'>> Loaded.')
    else:
        print(f'>> Generating new buffer dataset...')
        if dataset == 'FC100':
            X_train_clients, Y_train_clients, X_test, Y_test, client_class_cnt = \
                few_shot_partition(dataset, data_dir, 'noniid', n_clients, alpha=alpha)
        else:
            X_train, Y_train, X_test, Y_test, n_class = load_dataset(dataset, data_dir)

            train_data_by_class = rearrange_data_by_class(X_train, Y_train, n_class)

            X_train_clients, Y_train_clients, client_class_cnt = dirichlet_split_data(
                data_by_class=train_data_by_class,
                n_sample=len(X_train),
                n_class=n_class,
                n_clients=n_clients,
                alpha=alpha,
                sampling_ratio=sampling_ratio
            )

        buffer = {
            'X_train_clients': X_train_clients,
            'Y_train_clients': Y_train_clients,
            'X_test': X_test,
            'Y_test': Y_test,
            'client_class_cnt': client_class_cnt
        }

        if not os.path.isdir(data_dir + '/buffer'):
            os.makedirs(data_dir + '/buffer')

        torch.save(buffer, buffer_path)
        print(f">> New buffer dataset saved to {buffer_path}")

    return X_train_clients, Y_train_clients, X_test, Y_test, client_class_cnt


def dirichlet_split_data(data_by_class, n_sample, n_class, n_clients, alpha=0.5, sampling_ratio=1.0):
    """
    Split the train data into n_clients with dirichlet distribution - Dir(alpha)

    @param data_by_class: List of numpy array, each array contains all the data of the same class
    @param n_sample: Number of total samples in train data
    @param n_class: Number of classes
    @param n_clients: Number of clients
    @param alpha: Parameter for dirichlet distribution, the smaller the alpha, the more unbalanced the data
    @param sampling_ratio: The ratio of the data to be sampled from the total train data
    @return: X: list of clients' data, Y: list of clients' label, client_class_cnt: (client x class) count matrix
    """
    X = [[] for _ in range(n_clients)]
    Y = [[] for _ in range(n_clients)]
    client_class_cnt = [[0] * n_class for _ in range(n_clients)]
    samples_per_client = [0] * n_clients
    max_samples_per_client = int(sampling_ratio * n_sample / n_clients)
    for cls in range(n_class):
        # get indices for all that label
        idx_list = np.arange(len(data_by_class[cls]))
        np.random.shuffle(idx_list)
        if sampling_ratio < 1:
            samples_for_l = int(min(max_samples_per_client, int(sampling_ratio * len(data_by_class[cls]))))
            idx_list = idx_list[:samples_for_l]
        # dirichlet sampling from this label
        distribution = np.random.dirichlet(np.repeat(alpha, n_clients))
        while len(idx_list):
            # re-balance proportions to cap the number of samples for each user
            cumulative_distribution = np.cumsum(distribution)
            for c_id in range(n_clients - 1, -1, -1):
                if distribution[c_id] == 0: continue
                assign_size = min(max_samples_per_client - samples_per_client[c_id],
                                  int(distribution[c_id] / cumulative_distribution[c_id] * len(idx_list)))
                if assign_size == 0:
                    distribution[c_id] = 0
                    continue
                X[c_id].append(data_by_class[cls][idx_list[:assign_size]])
                Y[c_id].append(cls * np.ones(assign_size, dtype='int64'))
                client_class_cnt[c_id][cls] = assign_size
                samples_per_client[c_id] += assign_size
                idx_list = idx_list[assign_size:]
        # print(f'Class {cls} done')
        # print(f'Total {len(data_by_class[cls])}, Left {len(idx_list)}')
    # Convert data to numpy array
    for c_id in range(n_clients):
        X[c_id] = np.concatenate(X[c_id], axis=0)
        Y[c_id] = np.concatenate(Y[c_id], axis=0)
    return X, Y, np.array(client_class_cnt)


def few_shot_partition(dataset, data_dir, partition, n_clients, alpha=1.0, seed=0):
    if dataset == 'FC100':
        X_train, Y_train, X_test, Y_test, n_class = load_fc100_data(data_dir)
        train_classes = fine_split['train']
    else:
        raise ValueError('Unrecognized dataset')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


    client_data_idx = [[] for _ in range(n_clients)]
    class_client_cnt = [[0] * n_clients for _ in range(n_class)]
    if not alpha:
        for c in train_classes:
            c_idx = np.where(Y_train == c)[0]
            np.random.shuffle(c_idx)
            batch_idxs = np.array_split(c_idx, n_clients)
            for c_id in range(n_clients):
                client_data_idx[c_id].append(batch_idxs[c_id])
                class_client_cnt[c][c_id] = len(batch_idxs[c_id])
        for c_id in range(n_clients):
            client_data_idx[c_id] = list(np.concatenate(client_data_idx[c_id]))
            np.random.shuffle(client_data_idx[c_id])
    else:
        min_size = 0
        min_require_size = 10
        N = Y_train.shape[0]

        while min_size < min_require_size:
            client_data_idx = [[] for _ in range(n_clients)]
            for k in train_classes:
                idx_k = np.where(Y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
                proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, client_data_idx)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                class_client_cnt[k] = [len(one) for one in np.split(idx_k, proportions)]

                client_data_idx = [idx_j + idx.tolist() for idx_j, idx in zip(client_data_idx, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in client_data_idx])

        for j in range(n_clients):
            np.random.shuffle(client_data_idx[j])

    client_class_cnt = np.array(class_client_cnt).transpose()
    X_train_clients = [[] for _ in range(n_clients)]
    Y_train_clients = [[] for _ in range(n_clients)]
    for i, data_idx in enumerate(client_data_idx):
        X_train_clients[i] = X_train[data_idx]
        Y_train_clients[i] = Y_train[data_idx]

    return X_train_clients, Y_train_clients, X_test, Y_test, client_class_cnt


# There are 100 classes and 20 Superclasses in CIFAR100
fine_id_coarse_id = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18,
                     14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6,
                     26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9,
                     38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
                     50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3,
                     62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1,
                     74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19,
                     86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8,
                     98: 14, 99: 13}

# 6 : 2 : 2 split
coarse_split = {'train': [1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19],
                'valid': [8, 11, 13, 16], 'test': [0, 7, 12, 14]}

fine_split = defaultdict(list)
for fine_id, sparse_id in fine_id_coarse_id.items():
    if sparse_id in coarse_split['train']:
        fine_split['train'].append(fine_id)
    elif sparse_id in coarse_split['valid']:
        fine_split['valid'].append(fine_id)
    else:
        fine_split['test'].append(fine_id)