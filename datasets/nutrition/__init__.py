import os

import numpy as np
from torch.utils.data import DataLoader, Dataset

import settings

data_home = settings.DATA_HOME['nutrition']
dataset_file = os.path.join(data_home, 'nutrition_dataset.npz')


class NutritionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx],  self.labels[idx]

    def __len__(self):
        return len(self.labels)


def get_nutrition_data_loaders(batch_size):
    nutrition_dataset = np.load(dataset_file, allow_pickle=True)

    client_names = nutrition_dataset['client_names']
    all_data = nutrition_dataset['data']
    all_labels = nutrition_dataset['labels']

    np.random.seed(0)

    train_data_loaders = []
    test_data_loaders = []

    for client_name, data, labels in zip(client_names, all_data, all_labels):
        random_state = np.random.get_state()
        data = np.random.permutation(data)
        np.random.set_state(random_state)
        labels = np.random.permutation(labels)

        n_train = int(data.shape[0] * 0.9)
        train_data = data[:n_train]
        train_labels = labels[:n_train]
        test_data = data[n_train:]
        test_labels = labels[n_train:]

        n_0 = (train_labels == 0).sum()
        n_1 = n_train - n_0
        # print([train_data[train_labels == 1]] * (n_0 // n_1 + 1))
        train_syn_data = np.concatenate([train_data[train_labels == 1]] * (n_0 // n_1 + 1))[:n_0]
        train_syn_data = np.concatenate([train_data[train_labels == 0], train_syn_data])
        train_syn_labels = np.array([0] * n_0 + [1] * n_0)

        # print(data)
        mu = np.mean(train_syn_data)
        std = np.std(train_syn_data)
        train_syn_data = (train_syn_data - mu) / std
        train_data = (train_data - mu) / std
        test_data = (test_data - mu) / std

        train_data_loader = DataLoader(dataset=NutritionDataset(data=train_data, labels=train_labels),
                                       batch_size=batch_size, shuffle=True, num_workers=4)
        test_data_loader = DataLoader(dataset=NutritionDataset(data=test_data, labels=test_labels),
                                      batch_size=batch_size, shuffle=True, num_workers=4)

        train_data_loaders.append(train_data_loader)
        test_data_loaders.append(test_data_loader)

    return client_names, train_data_loaders, test_data_loaders


def get_nutrition_data_by_client(client_name):
    nutrition_dataset = np.load(dataset_file, allow_pickle=True)

    client_names = nutrition_dataset['client_names']
    all_data = nutrition_dataset['data']
    all_labels = nutrition_dataset['labels']

    client_idx = np.where(client_name == client_names)[0][0]

    return np.array(all_data[client_idx]), np.array(all_labels[client_idx])
