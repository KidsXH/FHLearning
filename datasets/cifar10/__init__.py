import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms as T

import settings

data_home = settings.DATA_HOME['cifar10']


class CIFARDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)


def get_cifar_data_loader(client_name, batch_size, split):
    dataset_file = os.path.join(data_home, '{}_dataset.npz'.format(client_name))
    dataset = np.load(dataset_file, allow_pickle=True)
    data = dataset['data']
    labels = dataset['labels']

    random_state = np.random.get_state()
    data = np.random.permutation(data)
    np.random.set_state(random_state)
    labels = np.random.permutation(labels)

    n_train = int(data.shape[0] * 0.9)
    train_data = data[:n_train]
    train_labels = labels[:n_train]
    test_data = data[n_train:]
    test_labels = labels[n_train:]

    # mu = np.mean(train_data)
    # std = np.std(train_data)
    # train_data = (train_data - mu) / std
    # test_data = (test_data - mu) / std

    transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFARDataset(data=train_data, labels=train_labels,
                                 transform=transform)
    test_dataset = CIFARDataset(data=test_data, labels=test_labels,
                                transform=transform)

    if split:
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_data_loader, test_data_loader
    else:
        data_loader = DataLoader(dataset=ConcatDataset([train_dataset, test_dataset]),
                                 batch_size=batch_size, shuffle=False, num_workers=4)
        return data_loader


def get_data_by_client(client_name):
    dataset_file = os.path.join(data_home, '{}_dataset.npz'.format(client_name))
    dataset = np.load(dataset_file, allow_pickle=True)
    data = dataset['data']
    labels = dataset['labels']
    new_data = np.array([np.transpose(image, (2, 0, 1)).reshape(-1) for image in data])
    return new_data, labels


def get_data_for_sampling(client_names):
    all_data = []
    all_labels = []
    for client_name in client_names:
        data, labels = get_data_by_client(client_name)
        all_data.append(data)
        all_labels.append(labels)
    ret = {'client_names': np.array(client_names),
           'data_shape': [3, 32, 32],
           'label_names': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
           'data': np.array(all_data),
           'labels': np.array(all_labels),
           'train_size': [5400] * len(client_names),
           'test_size': [600] * len(client_names),
           }
    return ret


def get_data_loader_by_samples(data, labels, batch_size=4):
    data = data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dateset = CIFARDataset(data, labels, transform)
    return DataLoader(dataset=dateset, batch_size=batch_size, shuffle=False, num_workers=4)
