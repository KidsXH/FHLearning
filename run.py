import numpy as np
import torch

from train.cifar10 import local_learning, local_predict, federated_learning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    print('Running on device:', device)
    client_names = ['Client-0', 'Client-3']
    local_learning(client_list=client_names, n_epochs=1)
    local_predict()
    federated_learning(communication_rounds=2, epochs_per_round=1, saving=True, max_num_parameters=1000)
