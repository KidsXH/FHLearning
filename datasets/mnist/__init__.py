import json

import torch
from torch.utils.data.dataset import T_co
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T
import numpy as np
import os
import settings

data_file = os.path.join(settings.DATA_HOME['mnist'], 'mnist_dataset.npz')
transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])


class MnistData(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        super(MnistData, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index) -> T_co:
        image, label = self.data[index], self.labels[index]
        image = self.transform(image).reshape(-1)
        return image, label

    def __len__(self):
        return self.data.shape[0]


def get_mnist_data_loaders(batch_size):
    """
    Get MNIST data.
    :param batch_size: batch size
    :return: client_names, train_loaders, test_loaders, server_data_loader
    """
    dataset = np.load(data_file, allow_pickle=True)
    all_data = dataset['data']
    all_labels = dataset['labels']
    client_names = dataset['client_names']
    train_loaders = []
    test_loaders = []
    np.random.seed(0)
    for data, labels in zip(all_data, all_labels):

        data = data.reshape((-1, 28, 28, 1))

        random_state = np.random.get_state()
        data = np.random.permutation(data)
        np.random.set_state(random_state)
        labels = np.random.permutation(labels)

        n_train = int(0.9 * len(data))
        train_data = data[:n_train]
        train_labels = labels[:n_train]
        test_data = data[n_train:]
        test_labels = labels[n_train:]

        train_dataset = MnistData(data=train_data, labels=train_labels, transform=transforms)
        train_loaders.append(DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4))
        test_dataset = MnistData(data=test_data, labels=test_labels, transform=transforms)
        test_loaders.append(DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4))

    train_loaders = np.array(train_loaders)
    test_loaders = np.array(test_loaders)

    return client_names, train_loaders, test_loaders


def new_data_loader(data, labels, batch_size=4):
    dateset = MnistData(data, labels, transform=transforms)
    return DataLoader(dataset=dateset, batch_size=batch_size, shuffle=False, num_workers=4)


def get_sample_data_loaders():
    samples_data = np.load(os.path.join(settings.DATA_HOME['mnist'], 'samples.npz'), allow_pickle=True)
    sampling_types = samples_data['sampling_types']
    client_list = samples_data['client_names']

    samples_data_loaders = {}
    for sampling_type in sampling_types:
        samples_data_loaders[sampling_type] = []
        for client_idx in range(client_list.shape[0]):
            data = samples_data[sampling_type][client_idx]
            data = data.reshape((-1, 28, 28, 1))
            labels = samples_data['ground_truth'][client_idx]
            data_loader = new_data_loader(data, labels)
            samples_data_loaders[sampling_type].append(data_loader)

    return client_list, sampling_types, samples_data_loaders


def get_data_for_sampling(client_list):
    dataset = np.load(data_file, allow_pickle=True)
    all_data = []
    all_labels = []
    for client_name, data, labels in zip(dataset['client_names'], dataset['data'], dataset['labels']):
        if client_name in client_list:
            all_data.append(data)
            all_labels.append(labels)

    ret = {'client_names': np.array(client_list),
           'data_shape': [28, 28],
           'label_names': ['Digit-0', 'Digit-1', 'Digit-2', 'Digit-3', 'Digit-4',
                           'Digit-5', 'Digit-6', 'Digit-7', 'Digit-8', 'Digit-9'],
           'data': np.array(all_data),
           'labels': np.array(all_labels),
           'train_size': [5400] * len(client_list),
           'test_size': [600] * len(client_list),
           }
    return ret
