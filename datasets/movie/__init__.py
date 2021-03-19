import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import T_co

import settings

vocab_counter_file = os.path.join(settings.DATA_HOME['movie'], 'vocab_counter.npy')
imdb_file = os.path.join(settings.DATA_HOME['movie'], 'imdb.npy')
amazon_file = os.path.join(settings.DATA_HOME['movie'], 'amazon.npy')

data_file = {'imdb': imdb_file,
             'amazon': amazon_file,
             }


class MovieData(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super(MovieData, self).__init__()

        self.data = data
        self.labels = labels

    def __getitem__(self, index) -> T_co:
        feat, label = self.data[index], self.labels[index]
        return feat, label

    def __len__(self):
        return self.data.shape[0]


def get_vocab_counter():
    counter = np.load(vocab_counter_file, allow_pickle=True)[()]
    return Counter(counter)


def get_data_loader(dataset, batch_size, percentage=0.9):
    all_data = np.load(data_file[dataset], allow_pickle=True)
    labels = all_data[:, 0]
    data = all_data[:, 1]

    data_neg = data[labels == 0]
    data_pos = data[labels == 1]
    labels_neg = labels[labels == 0]
    labels_pos = labels[labels == 1]

    n_neg = labels_neg.shape[0]
    n_train_neg = int(n_neg * percentage)
    n_pos = labels_neg.shape[0]
    n_train_pos = int(n_pos * percentage)

    train_data_neg, test_data_neg = data_neg[:n_train_neg], data_neg[n_train_neg:]
    train_data_pos, test_data_pos = data_pos[:n_train_pos], data_pos[n_train_pos:]
    train_labels_neg, test_labels_neg = labels_neg[:n_train_neg], labels_neg[n_train_neg:]
    train_labels_pos, test_labels_pos = labels_pos[:n_train_pos], labels_pos[n_train_pos:]

    train_data = np.concatenate((train_data_neg, train_data_pos))
    test_data = np.concatenate((test_data_neg, test_data_pos))
    train_labels = np.concatenate((train_labels_neg, train_labels_pos))
    test_labels = np.concatenate((test_labels_neg, test_labels_pos))

    train_dataset = MovieData(train_data, train_labels)
    test_dataset = MovieData(test_data, test_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, test_loader
