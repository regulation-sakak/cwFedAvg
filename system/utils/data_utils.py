import numpy as np
import os
import torch

DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'dataset')

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(DATA_PATH, dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(DATA_PATH, dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if dataset[:2] == "News":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "news":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "Shakespeare":
        return read_client_data_Shakespeare(dataset, idx)

    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

