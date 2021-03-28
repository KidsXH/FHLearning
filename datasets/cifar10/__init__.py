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


train_transforms = T.Compose([T.ToTensor(),
                              T.Pad(4, padding_mode='reflect'),
                              T.RandomHorizontalFlip(),
                              T.RandomCrop(32),
                              T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
val_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_cifar_data_loader(client_name, batch_size):
    dataset_file = os.path.join(data_home, '{}_dataset.npz'.format(client_name))
    dataset = np.load(dataset_file, allow_pickle=True)
    data = dataset['data']
    labels = dataset['labels']

    n_train = 5000
    train_data = data[:n_train]
    train_labels = labels[:n_train]
    test_data = data[n_train:]
    test_labels = labels[n_train:]

    train_dataset = CIFARDataset(data=train_data, labels=train_labels, transform=train_transforms)
    test_dataset = CIFARDataset(data=test_data, labels=test_labels, transform=val_transforms)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_data_loader, test_data_loader


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
           'label_names': ['plane', 'car', 'ship', 'truck'],
           'data': np.array(all_data),
           'labels': np.array(all_labels),
           'train_size': [4500] * len(client_names),
           'test_size': [900] * len(client_names),
           }
    return ret


def new_data_loader(data, labels, batch_size=4):
    data = data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    dateset = CIFARDataset(data, labels, val_transforms)
    return DataLoader(dataset=dateset, batch_size=batch_size, shuffle=False, num_workers=4)


def get_sample_data_loaders():
    samples_data = np.load(os.path.join(settings.DATA_HOME['cifar10'], 'samples.npz'), allow_pickle=True)
    sampling_types = samples_data['sampling_types']
    client_list = samples_data['client_names']

    samples_data_loaders = {}
    for sampling_type in sampling_types:
        samples_data_loaders[sampling_type] = []
        for client_idx in range(client_list.shape[0]):
            data = samples_data[sampling_type][client_idx]
            labels = samples_data['ground_truth'][client_idx]
            data_loader = new_data_loader(data, labels)
            samples_data_loaders[sampling_type].append(data_loader)

    return client_list, sampling_types, samples_data_loaders
