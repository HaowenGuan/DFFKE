import os
import torch
import numpy as np
from dataset.utils_dataset import DataDistributor


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('data', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('data', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    data = read_data(dataset, idx, is_train)
    X = torch.Tensor(data['x']).type(torch.float32)
    Y = torch.Tensor(data['y']).type(torch.int64)
    data = [(x, y) for x, y in zip(X, Y)]
    return data


def read_client_data_custom(data_distributor: DataDistributor, idx, is_train=True):

    if is_train:
        X, Y = data_distributor.get_client_train_data(idx)
        X_train = torch.Tensor(X).type(torch.float32)
        y_train = torch.Tensor(Y).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        X, Y = data_distributor.get_client_test_data(idx)
        X_test = torch.Tensor(X).type(torch.float32)
        y_test = torch.Tensor(Y).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
