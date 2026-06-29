import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, DatasetFolder, ImageFolder

from dataset.transforms import get_cifar10_transform, get_cifar100_transform, get_tiny_imagenet_transform


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

    def update(self, other_data: np.array, other_target: np.array):
        if len(self.data) == 0:
            self.data = other_data
            self.target = other_target
        else:
            self.data = np.concatenate((self.data, other_data))
            self.target = np.concatenate((self.target, other_target))


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
    def __init__(self, fake_data, emb, logit, target, transform=None):
        self.fake_data = np.array(fake_data)
        self.emb = np.array(emb)
        self.logit = np.array(logit)
        self.target = np.array(target)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (fake_data, emb, logit) where target is index of the target class.
        """
        fake_data, emb, logit, target = self.fake_data[index], self.emb[index], self.logit[index], self.target[index]

        if self.transform is not None:
            fake_data = self.transform(fake_data)

        return fake_data, emb, logit, target

    def __len__(self):
        return len(self.fake_data)

    def update(self, other_fake_dataset: 'FakeDataset'):
        self.fake_data = np.concatenate((self.fake_data, other_fake_dataset.fake_data))
        self.emb = np.concatenate((self.emb, other_fake_dataset.emb))
        self.logit = np.concatenate((self.logit, other_fake_dataset.logit))
        self.target = np.concatenate((self.target, other_fake_dataset.target))


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


class DataDistributor:
    def __init__(self, args):
        X_train_clients, Y_train_clients, X_test_list, Y_test_list, client_class_cnt = get_federated_learning_dataset(
            dataset=args['dataset'],
            data_dir=args['data_dir'],
            n_clients=args['n_clients'],
            partition_mode=args['data_partition'],
            alpha=args['alpha'],
            redo_split=args['redo_ds_split']
        )
        X_train, Y_train, X_test, Y_test, n_class, img_size = load_dataset(args['dataset'], args['data_dir'])

        self.is_PFL = 'personalized' in args['data_partition']
        self.client_class_cnt = client_class_cnt
        self.n_class = n_class
        self.img_size = img_size
        self.client_label_list = [[] for _ in range(args['n_clients'])]
        for i, l in enumerate(self.client_class_cnt):
            self.client_label_list[i] = [cls for cls, cnt in enumerate(l) if cnt > 0]

        if args['dataset'] == 'CIFAR10':
            transform = get_cifar10_transform(args['resize_img'])
        elif args['dataset'] in ['CIFAR100', 'FC100']:
            transform = get_cifar100_transform(args['resize_img'])
        elif args['dataset'] == 'TinyImageNet':
            transform = get_tiny_imagenet_transform(args['resize_img'])
        else:
            raise ValueError('Unknown encoder')
        train_transform = transform['train_transform']
        test_transform = transform['test_transform']

        self.client_train_loaders = []
        self.non_aug_train_loaders = []
        self.client_test_loaders = []
        for client_id in range(args['n_clients']):
            # Get the private data for the client
            X_train_client = X_train_clients[client_id]
            Y_train_client = Y_train_clients[client_id]
            print(f'>> Client {client_id} owns {len(X_train_client)} training samples.')
            print('Train:', sorted(self.client_class_cnt[client_id], reverse=True))
            # print(sorted([[cnt, i] for i, cnt in enumerate(client_class_cnt[client_id])], reverse=True))

            # Create the private data loader for each client
            train_dataset = CustomDataset(X_train_client, Y_train_client, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
            self.client_train_loaders.append(train_loader)

            # Create the non aug data loader for each client
            train_dataset = CustomDataset(X_train_client, Y_train_client, transform=test_transform)
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
            self.non_aug_train_loaders.append(train_loader)

            if self.is_PFL:
                # Create the test data loader for each client
                X_test_client = X_test_list[client_id]
                Y_test_client = Y_test_list[client_id]
                test_dataset = CustomDataset(X_test_client, Y_test_client, transform=test_transform)
                test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)
                self.client_test_loaders.append(test_loader)

        # For visualization only
        self.client_visual_loaders = None
        if 'visualize_decoder_output' in args and args['visualize_decoder_output']:
            self.client_visual_loaders = dataset_visualization(
                args=args,
                X_train_clients=X_train_clients,
                Y_train_clients=Y_train_clients,
                n_class=self.n_class,
                non_aug_transform=test_transform
            )

        # full train set data loader for evaluation purpose
        print(f'>> Full Train Set Size: {len(X_train)}')
        train_dataset = CustomDataset(X_train, Y_train, transform=test_transform)
        self.full_train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)

        # full test set data loader for evaluation purpose
        print(f'>> Full Test Set Size: {len(X_test)}')
        test_dataset = CustomDataset(X_test, Y_test, transform=test_transform)
        self.full_test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)

        torch.multiprocessing.set_sharing_strategy('file_system')
        self.X_train_clients = []
        self.Y_train_clients = []
        for loader in self.client_train_loaders:
            X = []
            Y = []
            for x, y in loader:
                X.append(x.detach().cpu())
                Y.append(y.detach().cpu())
            X = torch.cat(X, dim=0)
            Y = torch.cat(Y, dim=0)
            self.X_train_clients.append(X)
            self.Y_train_clients.append(Y)

        if self.is_PFL:
            self.X_test_clients = []
            self.Y_test_clients = []
            for loader in self.client_test_loaders:
                X = []
                Y = []
                for x, y in loader:
                    X.append(x.detach().cpu())
                    Y.append(y.detach().cpu())
                X = torch.cat(X, dim=0)
                Y = torch.cat(Y, dim=0)
                self.X_test_clients.append(X)
                self.Y_test_clients.append(Y)

        self.X_test = []
        self.Y_test = []
        for x, y in self.full_test_loader:
            self.X_test.append(x.detach().cpu())
            self.Y_test.append(y.detach().cpu())
        self.X_test = torch.cat(self.X_test, dim=0)
        self.Y_test = torch.cat(self.Y_test, dim=0)

    def get_client_train_data(self, client_id):
        return self.X_train_clients[client_id], self.Y_train_clients[client_id]

    def get_client_test_data(self, client_id):
        if self.is_PFL:
            return self.X_test_clients[client_id], self.Y_test_clients[client_id]
        return self.X_test, self.Y_test

    def get_client_train_loader(self, client_id):
        return self.client_train_loaders[client_id]

    def get_client_test_loader(self, client_id):
        if self.is_PFL:
            return self.client_test_loaders[client_id]
        return self.full_test_loader

    def get_client_label_list(self, client_id):
        return self.client_label_list[client_id]

    def get_train_loader(self):
        return self.full_train_loader

    def get_test_loader(self):
        return self.full_test_loader


def load_cifar10_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train = CIFAR10(data_dir, train=True, transform=transform, download=True)
    cifar10_test = CIFAR10(data_dir, train=False, transform=transform, download=True)

    X_train, Y_train = cifar10_train.data, np.array(cifar10_train.targets)
    X_test, Y_test = cifar10_test.data, np.array(cifar10_test.targets)
    n_class = len(cifar10_train.classes)
    img_size = 32

    return X_train, Y_train, X_test, Y_test, n_class, img_size


def load_cifar100_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train = CIFAR100(data_dir, train=True, transform=transform, download=True)
    cifar100_test = CIFAR100(data_dir, train=False, transform=transform, download=True)

    X_train, Y_train = cifar100_train.data, np.array(cifar100_train.targets)
    X_test, Y_test = cifar100_test.data, np.array(cifar100_test.targets)
    n_class = len(cifar100_train.classes)
    img_size = 32

    return X_train, Y_train, X_test, Y_test, n_class, img_size


def load_tiny_imagenet_data(data_dir):
    if not os.path.exists(os.path.join(data_dir, 'tiny-imagenet-200')):
        os.system(f'wget --directory-prefix {data_dir} http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        file_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')
        os.system(f'unzip {file_path} -d {data_dir}')

        # Tiny ImageNet reformat
        # ==================================================================================================
        import glob
        from shutil import move
        target_folder = '../data/tiny-imagenet-200/val/'

        val_dict = {}
        with open(target_folder + 'val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]

        paths = glob.glob(target_folder + 'images/*')
        paths[0].split('/')[-1]
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(target_folder + str(folder)):
                os.mkdir(target_folder + str(folder))

        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            dest = target_folder + str(folder) + '/' + str(file)
            move(path, dest)

        os.remove('../data/tiny-imagenet-200/val/val_annotations.txt')
        os.rmdir('../data/tiny-imagenet-200/val/images')
        print('done reformat the validation images')
        # ==================================================================================================
    else:
        print('Files already downloaded and verified\n')

    transform = transforms.Compose([lambda x: np.array(x)])

    train_set = ImageFolder_custom(root=os.path.join(data_dir, 'tiny-imagenet-200/train/'), transform=transform)
    test_set = ImageFolder_custom(root=os.path.join(data_dir, 'tiny-imagenet-200/val/'), transform=transform)
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=len(train_set), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=len(test_set), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        train_set.data, train_set.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        test_set.data, test_set.targets = test_data

    X_train, Y_train = np.array(train_set.data), np.array(train_set.targets)
    X_test, Y_test = np.array(test_set.data), np.array(test_set.targets)
    n_class = 200
    img_size = 64

    return X_train, Y_train, X_test, Y_test, n_class, img_size


# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


def load_fc100_data(data_dir):
    X_train, Y_train, X_test, Y_test, n_class = load_cifar100_data(data_dir)

    X_total = np.concatenate([X_train, X_test], 0)
    y_total = np.concatenate([Y_train, Y_test], 0)

    test_data_idxs = []
    for cls in fine_split['test']:
        test_data_idxs.extend(np.where(y_total == cls)[0].tolist())

    X_test = X_total[test_data_idxs]
    Y_test = y_total[test_data_idxs]

    train_data_idx = []
    for cls in fine_split['train']:
        train_data_idx.extend(np.where(y_total == cls)[0].tolist())

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
    elif dataset == 'TinyImageNet':
        print(f"Loading TinyImageNet data...")
        return load_tiny_imagenet_data(data_dir)
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")


def rearrange_data_by_class(data, targets, class_list):
    """
    Rearrange the data by class
    :param data: numpy array of data
    :param targets: numpy array of targets
    :param class_list: list of classes
    :return: dictionary of data by class
    """
    new_data = dict()
    for cls in class_list:
        idx = targets == cls
        new_data[cls] = data[idx]
    return new_data


def get_federated_learning_dataset(
        dataset,
        data_dir,
        n_clients,
        partition_mode,
        alpha=1.0,
        sampling_ratio=1.0,
        redo_split=False):
    """
    Get the federated learning dataset by partitioning the data into multiple clients

    :param dataset: Dataset name, (CIFAR10, CIFAR100, FC100)
    :param data_dir: Directory to save the data
    :param n_clients: Number of clients
    :param partition_mode: (iid, non_iid_unbalanced, non_iid_balanced)
    :param alpha: Degree of non-iid data split (0.1 High Hetero, 1.0 Medium Hetero, 10.0 Low Hetero)
    :param sampling_ratio: Ratio of training data to be used
    :param redo_split: If True, overwrite existing split cache
    :return: X_train_clients, Y_train_clients, X_test, Y_test, client_class_cnt
    """
    cache_path = data_dir + f'cache/{dataset}_{n_clients}client_{partition_mode}'
    if partition_mode != 'iid':
        print(f'>> Split data for FL with Dir({float(alpha)})')
        cache_path += f'_{float(alpha)}alpha'
    cache_path += f'_{float(sampling_ratio)}ratio.pt'

    if os.path.exists(cache_path) and not redo_split:
        print(f'>> Loading buffer dataset from {cache_path}')
        cache = torch.load(cache_path)
        X_train_clients = [np.array(x).copy() for x in cache['X_train_clients']]
        Y_train_clients = [np.array(y).copy() for y in cache['Y_train_clients']]
        X_test = [np.array(x).copy() for x in cache['X_test']]
        Y_test = [np.array(y).copy() for y in cache['Y_test']]
        client_class_cnt = np.array(cache['client_class_cnt']).copy()
        del cache
        print(f'>> Loaded.')
    else:
        print(f'Cache file {cache_path} not found.')
        print(f'>> Generating new cache dataset...')
        X_train, Y_train, X_test, Y_test, n_class, img_size = load_dataset(dataset, data_dir)
        if dataset == 'FC100':
            class_list = fine_split['train']
        else:
            class_list = list(range(n_class))

        train_data_by_class = rearrange_data_by_class(X_train, Y_train, class_list)
        test_data_by_class = rearrange_data_by_class(X_test, Y_test, class_list)

        if partition_mode == 'iid':
            X_train_clients, Y_train_clients, client_class_cnt = iid_partition(
                data_by_class=train_data_by_class,
                n_class=n_class,
                n_clients=n_clients,
            )
        elif partition_mode.startswith('non_iid'):
            X_train_clients, Y_train_clients, client_class_cnt = non_iid_dirichlet_partition(
                data_by_class=train_data_by_class,
                n_sample=len(X_train),
                n_class=n_class,
                n_clients=n_clients,
                alpha=alpha,
                balance='balanced' in partition_mode,
                sample_ratio=sampling_ratio
            )
        elif partition_mode.startswith('personalized_non_iid'):
            X_train_clients, Y_train_clients, X_test, Y_test, client_class_cnt = non_iid_dirichlet_personal(
                train=train_data_by_class,
                test=test_data_by_class,
                n_sample=len(X_test),
                n_class=n_class,
                n_clients=n_clients,
                alpha=alpha,
                balance='balanced' in partition_mode,
                sample_ratio=sampling_ratio
            )
        elif partition_mode.startswith('personalized_pathological'):
            X_train_clients, Y_train_clients, X_test, Y_test, client_class_cnt = non_iid_pathological_personal(
                train=train_data_by_class,
                test=test_data_by_class,
                n_sample=len(X_test),
                n_class=n_class,
                n_clients=n_clients,
                alpha=alpha,
                balance='balanced' in partition_mode,
                sample_ratio=sampling_ratio
            )
        else:
            raise ValueError(f"Partition mode {partition_mode} is not supported yet.")

        cache = {
            'X_train_clients': X_train_clients,
            'Y_train_clients': Y_train_clients,
            'X_test': X_test,
            'Y_test': Y_test,
            'client_class_cnt': client_class_cnt
        }

        if not os.path.isdir(data_dir + '/cache'):
            os.makedirs(data_dir + '/cache')

        torch.save(cache, cache_path)
        print(f">> New cache dataset saved to {cache_path}")

    return X_train_clients, Y_train_clients, X_test, Y_test, client_class_cnt


def iid_partition(data_by_class, n_class, n_clients):
    """
    Split the train data into n_clients with iid distribution

    :param data_by_class: Dict of numpy array, each array contains all the data of one class
    :param n_class: Number of classes
    :param n_clients: Number of clients
    :return: X: list of clients' data, Y: list of clients' label, client_class_cnt: (client x class) count matrix
    """
    X = [[] for _ in range(n_clients)]
    Y = [[] for _ in range(n_clients)]
    client_class_cnt = np.zeros((n_clients, n_class), dtype='int64')
    for cls in tqdm(data_by_class):
        idx_list = np.arange(len(data_by_class[cls]))
        np.random.shuffle(idx_list)
        split_idx_list = np.array_split(idx_list, n_clients)
        for c_id, c_idx_list in enumerate(split_idx_list):
            X[c_id].append(data_by_class[cls][c_idx_list])
            Y[c_id].append(cls * np.ones(len(c_idx_list), dtype='int64'))
            client_class_cnt[c_id][cls] += len(c_idx_list)
    # Convert data to numpy array and return
    return [np.concatenate(l) for l in X], [np.concatenate(l) for l in Y], client_class_cnt


def non_iid_dirichlet_partition(data_by_class, n_sample, n_class, n_clients, alpha=1.0, balance=True, sample_ratio=1.0):
    """
    Split the train data into n_clients with dirichlet distribution - Dir(alpha)
    If Balance, balance the data distribution so each client has the same number of samples

    @param data_by_class: Dict of numpy array, each array contains all the data of one class
    @param n_sample: Number of total samples in train data
    @param n_class: Number of classes
    @param n_clients: Number of clients
    @param alpha: Parameter for dirichlet distribution, the smaller the alpha, the more unbalanced the data
    @param balance: If True, balance the data distribution so each client has the same number of samples
    @param sample_ratio: The ratio of the data to be sampled from the total train data
    @return: X: list of clients' data, Y: list of clients' label, client_class_cnt: (client x class) count matrix
    """
    X = [[] for _ in range(n_clients)]
    Y = [[] for _ in range(n_clients)]
    client_class_cnt = np.zeros((n_clients, n_class), dtype='int64')
    samples_per_client = [0] * n_clients
    limit = int(sample_ratio * n_sample / n_clients)
    for cls in tqdm(list(data_by_class)):
        # get indices for all that label
        idx_list = np.arange(len(data_by_class[cls]))
        np.random.shuffle(idx_list)
        if sample_ratio < 1:
            samples_for_l = int(min(limit, int(sample_ratio * len(data_by_class[cls]))))
            idx_list = idx_list[:samples_for_l]
        # dirichlet sampling from this label
        distribution = np.random.dirichlet(np.repeat(alpha, n_clients))
        if balance:
            while len(idx_list):
                # re-balance proportions to cap the number of samples for each user
                cumulative_distribution = np.cumsum(distribution)
                for c_id in range(n_clients - 1, -1, -1):
                    if distribution[c_id] == 0: continue
                    assign_size = min(limit - samples_per_client[c_id],
                                      int(distribution[c_id] / cumulative_distribution[c_id] * len(idx_list)))
                    if samples_per_client[c_id] == limit:
                        distribution[c_id] = 0
                        continue
                    X[c_id].append(data_by_class[cls][idx_list[:assign_size]])
                    Y[c_id].append(cls * np.ones(assign_size, dtype='int64'))
                    client_class_cnt[c_id][cls] += assign_size
                    samples_per_client[c_id] += assign_size
                    idx_list = idx_list[assign_size:]
        else:
            distribution = np.array([p * (cnt < limit) for p, cnt in zip(distribution, samples_per_client)])
            distribution = distribution / distribution.sum()
            distribution = (np.cumsum(distribution) * len(idx_list)).astype(int)[:-1]
            for c_id, c_idx_list in enumerate(np.split(idx_list, distribution)):
                X[c_id].append(data_by_class[cls][c_idx_list])
                Y[c_id].append(cls * np.ones(len(c_idx_list), dtype='int64'))
                client_class_cnt[c_id][cls] += len(c_idx_list)
                samples_per_client[c_id] += len(c_idx_list)
    # Convert data to numpy array and return
    return [np.concatenate(l) for l in X], [np.concatenate(l) for l in Y], client_class_cnt


def non_iid_dirichlet_personal(train, test, n_sample, n_class, n_clients, alpha=1.0, balance=True, sample_ratio=1.0):
    """
    Split the train and test data into n_clients with dirichlet distribution - Dir(alpha)
    If Balance, balance the data distribution so each client has the same number of samples
    Note:
    Distinctively, this function splits the test data, and construct the train data based on the test data distribution.
    In this way, the test data and train data will have the same distribution.

    @param train: Dict of numpy array, data_by_class, each array contains all the train data of one class
    @param test: Dict of numpy array, data_by_class, each array contains all the test data of one class
    @param n_sample: Number of total samples in [test] data
    @param n_class: Number of classes
    @param n_clients: Number of clients
    @param alpha: Parameter for dirichlet distribution, the smaller the alpha, the more unbalanced the data
    @param balance: If True, balance the data distribution so each client has the same number of samples
    @param sample_ratio: The ratio of the data to be sampled from the total train data
    @return: X: list of clients' data, Y: list of clients' label, client_class_cnt: (client x class) count matrix
    """
    train_test_ratio = len(train[0]) / len(test[0])
    X_train = [[] for _ in range(n_clients)]
    Y_train = [[] for _ in range(n_clients)]
    X_test = [[] for _ in range(n_clients)]
    Y_test = [[] for _ in range(n_clients)]
    client_class_cnt = np.zeros((n_clients, n_class), dtype='int64')
    samples_per_client = [0] * n_clients
    limit = int(sample_ratio * n_sample / n_clients)
    for cls in tqdm(list(test)):
        # get indices for all that label
        idx_list = np.arange(len(test[cls]))
        train_idx_list = np.arange(len(train[cls]))
        np.random.shuffle(idx_list)
        np.random.shuffle(train_idx_list)
        if sample_ratio < 1:
            samples_for_l = int(min(limit, int(sample_ratio * len(test[cls]))))
            idx_list = idx_list[:samples_for_l]
        # dirichlet sampling from this label
        distribution = np.random.dirichlet(np.repeat(alpha, n_clients))
        if balance:
            while len(idx_list):
                # re-balance proportions to cap the number of samples for each user
                cumulative_distribution = np.cumsum(distribution)
                for c_id in range(n_clients - 1, -1, -1):
                    if distribution[c_id] == 0: continue
                    assign_size = min(limit - samples_per_client[c_id],
                                      int(distribution[c_id] / cumulative_distribution[c_id] * len(idx_list)))
                    if samples_per_client[c_id] == limit:
                        distribution[c_id] = 0
                        continue
                    # Assign test data
                    X_test[c_id].append(test[cls][idx_list[:assign_size]])
                    Y_test[c_id].append(cls * np.ones(assign_size, dtype='int64'))
                    idx_list = idx_list[assign_size:]
                    samples_per_client[c_id] += assign_size
                    # Assign train data
                    train_assign_size = int(train_test_ratio * assign_size)
                    X_train[c_id].append(train[cls][train_idx_list[:train_assign_size]])
                    Y_train[c_id].append(cls * np.ones(train_assign_size, dtype='int64'))
                    train_idx_list = train_idx_list[train_assign_size:]
                    client_class_cnt[c_id][cls] += train_assign_size

        else:
            distribution = np.array([p * (cnt < limit) for p, cnt in zip(distribution, samples_per_client)])
            distribution = distribution / distribution.sum()
            distribution = (np.cumsum(distribution) * len(idx_list)).astype(int)[:-1]
            for c_id, c_idx_list in enumerate(np.split(idx_list, distribution)):
                X_test[c_id].append(test[cls][c_idx_list])
                Y_test[c_id].append(cls * np.ones(len(c_idx_list), dtype='int64'))
                samples_per_client[c_id] += len(c_idx_list)
                # Assign train data
                train_assign_size = int(train_test_ratio * len(c_idx_list))
                X_train[c_id].append(train[cls][train_idx_list[:train_assign_size]])
                Y_train[c_id].append(cls * np.ones(train_assign_size, dtype='int64'))
                train_idx_list = train_idx_list[train_assign_size:]
                client_class_cnt[c_id][cls] += train_assign_size
    # Convert data to numpy array and return
    X_train = [np.concatenate(client_list) for client_list in X_train]
    Y_train = [np.concatenate(client_list) for client_list in Y_train]
    X_test = [np.concatenate(client_list) for client_list in X_test]
    Y_test = [np.concatenate(client_list) for client_list in Y_test]
    return X_train, Y_train, X_test, Y_test, client_class_cnt


def non_iid_pathological_personal(train, test, n_sample, n_class, n_clients, alpha=1.0, balance=True, sample_ratio=1.0):
    """
    Split the train and test data into n_clients with dirichlet distribution - Dir(alpha)
    If Balance, balance the data distribution so each client has the same number of samples
    Note:
    Distinctively, this function splits the test data, and construct the train data based on the test data distribution.
    In this way, the test data and train data will have the same distribution.

    @param train: Dict of numpy array, data_by_class, each array contains all the train data of one class
    @param test: Dict of numpy array, data_by_class, each array contains all the test data of one class
    @param n_sample: Number of total samples in [test] data
    @param n_class: Number of classes
    @param n_clients: Number of clients
    @param alpha: Parameter for dirichlet distribution, the smaller the alpha, the more unbalanced the data
    @param balance: If True, balance the data distribution so each client has the same number of samples
    @param sample_ratio: The ratio of the data to be sampled from the total train data
    @return: X: list of clients' data, Y: list of clients' label, client_class_cnt: (client x class) count matrix
    """
    n_sample_per_class = 0
    if n_class == 10:
        if n_clients < 10:
            raise ValueError('Pathological partitioning is only supported for CIFAR10 with more than 10 clients.')
        n_sample_per_class = int(5000 / n_clients)  # 2500 train and 500 test samples per class, if n_clients == 10
    if n_class == 100:
        if n_clients < 10:
            raise ValueError('Pathological partitioning is only supported for CIFAR100 with more than 10 clients.')
        n_sample_per_class = int(1000 / n_clients)  # 500 train and 100 test samples per class, if n_clients == 100
    if not balance:
        raise ValueError('Pathological partitioning is only supported for balanced data distribution.')
    train_test_ratio = len(train[0]) / len(test[0])
    X_train = [[] for _ in range(n_clients)]
    Y_train = [[] for _ in range(n_clients)]
    X_test = [[] for _ in range(n_clients)]
    Y_test = [[] for _ in range(n_clients)]
    client_class_cnt = np.zeros((n_clients, n_class), dtype='int64')
    samples_per_client = [0] * n_clients
    limit = int(sample_ratio * n_sample / n_clients)
    for cls in tqdm(list(test)):
        # get indices for all that label
        idx_list = np.arange(len(test[cls]))
        train_idx_list = np.arange(len(train[cls]))
        np.random.shuffle(idx_list)
        np.random.shuffle(train_idx_list)
        if sample_ratio < 1:
            samples_for_l = int(min(limit, int(sample_ratio * len(test[cls]))))
            idx_list = idx_list[:samples_for_l]
        # dirichlet sampling from this label
        order = np.random.permutation(n_clients).tolist()
        if balance:
            while len(idx_list):
                c_id = order.pop()
                if samples_per_client[c_id] == limit:
                    continue
                assign_size = n_sample_per_class
                # Assign test data
                X_test[c_id].append(test[cls][idx_list[:assign_size]])
                Y_test[c_id].append(cls * np.ones(assign_size, dtype='int64'))
                idx_list = idx_list[assign_size:]
                samples_per_client[c_id] += assign_size
                # Assign train data
                train_assign_size = int(train_test_ratio * assign_size)
                X_train[c_id].append(train[cls][train_idx_list[:train_assign_size]])
                Y_train[c_id].append(cls * np.ones(train_assign_size, dtype='int64'))
                train_idx_list = train_idx_list[train_assign_size:]
                client_class_cnt[c_id][cls] += train_assign_size

        else:
            raise NotImplementedError
    # Convert data to numpy array and return
    X_train = [np.concatenate(client_list) for client_list in X_train]
    Y_train = [np.concatenate(client_list) for client_list in Y_train]
    X_test = [np.concatenate(client_list) for client_list in X_test]
    Y_test = [np.concatenate(client_list) for client_list in Y_test]
    return X_train, Y_train, X_test, Y_test, client_class_cnt


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def dataset_visualization(args, X_train_clients, Y_train_clients, n_class, non_aug_transform):
    if args['dataset'] != 'CIFAR100':
        raise ValueError('Visualization is only supported for CIFAR100')

    client_visual_loaders = []
    for client_id in range(args['n_clients']):
        X_train_client = X_train_clients[client_id]
        Y_train_client = Y_train_clients[client_id]
        # get one sample per class for each client for visualization
        X_client_visualization = []
        Y_client_visualization = []
        seen = [False] * n_class
        for x, y in zip(X_train_client, Y_train_client):
            if not seen[y]:
                X_client_visualization.append(x)
                Y_client_visualization.append(y)
                seen[y] = True
        visual_dataset = CustomDataset(X_client_visualization, Y_client_visualization, transform=non_aug_transform)
        client_visual_loaders.append(DataLoader(visual_dataset, batch_size=1, shuffle=False))

    # Save the visualization data
    device = args['device']
    visual_folder = os.path.join(args['checkpoint_dir'], 'synthetic_data_visualization')
    mkdir(visual_folder)
    visual_folder = os.path.join(visual_folder, f'original_data')
    mkdir(visual_folder)
    # Mean and std for CIFAR100
    mean = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).to(device)
    std = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).to(device)
    for i in range(args['n_clients']):
        client_visual_folder = os.path.join(visual_folder, f'client_{i}')
        mkdir(client_visual_folder)
        visual_loader = client_visual_loaders[i]
        for x, y in visual_loader:
            x, y = x.to(device), y.to(device)
            # Save to folder
            fake_data = x[0] * std[:, None, None] + mean[:, None, None]
            fake_data = (fake_data * 255).clamp(0, 255).byte()
            image = transforms.ToPILImage()(fake_data.permute(0, 1, 2))
            image.save(os.path.join(client_visual_folder, f'{i}_{y.item()}_ori.png'))
    print(f'>> Visualized original data saved to {visual_folder}')
    return client_visual_loaders


########################################## CIFAR100 Few Shot Partition ##########################################
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
#################################################################################################################
