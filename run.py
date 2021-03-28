import numpy as np
import torch

from models.cifar10 import CIFARModel
from train.cifar10 import local_learning, local_predict, federated_learning
from utils import set_random_seed, get_sample_idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    print('Running on device:', device)

    sampling_idx_layers, sampling_idx_all = get_sample_idx(CIFARModel(), max_num_parameters=1000)

    set_random_seed(0)
    client_names = ['Client-0', 'Client-3']
    local_learning(client_list=client_names, n_epochs=1)
    local_predict()
    federated_learning(communication_rounds=2, epochs_per_round=1, saving=True,
                       sampling_idx_layers=sampling_idx_layers, sampling_idx_all=sampling_idx_all)
