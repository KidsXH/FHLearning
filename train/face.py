import os
from time import time

import numpy as np
import torch

import settings
from datasets.face import get_face_data_loader, get_sample_data_loaders
from models.face import Client, FaceModel, Server
from utils import parameters2weights, cos_v

batch_size = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def local_learning(client_list, n_epochs):
    train_loaders = []
    test_loaders = []
    for client_name in client_list:
        train_loader, test_loader = get_face_data_loader(client_name, batch_size=batch_size)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    local_clients = []

    for client_name, train_loader, test_loader in zip(client_list, train_loaders, test_loaders):
        local_clients.append(Client(client_name=client_name + '_Local',
                                    train_loader=train_loader, test_loader=test_loader,
                                    checkpoint_path=os.path.join(settings.CHECKPOINTS_DIR, client_name),
                                    device=device))

    for client in local_clients:
        print('Training:', client.name)
        for epoch in range(n_epochs):
            client.run(n_epochs=1, save_last=True)
            client.test(save_best=True)


def local_predict():
    client_names, sampling_types, samples_data_loaders = get_sample_data_loaders()

    for client_idx, client_name in enumerate(client_names):
        checkpoint_path = os.path.join(settings.CHECKPOINTS_DIR, client_name, '{}_Local_last.cp'.format(client_name))
        model = FaceModel().to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        results = {}
        for sampling_type in sampling_types:
            print('Predicting {} {}'.format(client_name, sampling_type))
            data_loader = samples_data_loaders[sampling_type][client_idx]
            predictions = []
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs = inputs.float().to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.append(predicted.cpu().numpy())
            results[sampling_type] = np.concatenate(predictions)
        output_file = os.path.join(settings.OUTPUTS_DIR, '{}_local'.format(client_name))
        np.savez_compressed(output_file, **results)


def federated_learning(communication_rounds=1, epochs_per_round=1, saving=False, n_sampling_parameters=1000):
    client_list, sampling_types, samples_data_loaders = get_sample_data_loaders()
    client_names = client_list
    train_loaders = []
    test_loaders = []
    for client_name in client_names:
        train_loader, test_loader = get_face_data_loader(client_name, batch_size=batch_size)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    # Initiate Parameters
    server = Server(start_round=0, checkpoint_path=os.path.join(settings.CHECKPOINTS_DIR, 'Server'), device=device)

    n_paras = sum(p.numel() for p in server.model.parameters())
    print('n_paras: {}'.format(n_paras))

    np.random.seed(0)
    sampling_idx = np.random.permutation(n_paras)[:n_sampling_parameters]

    federated_clients = []

    for client_name, train_loader, test_loader in zip(client_names, train_loaders, test_loaders):
        federated_clients.append(Client(client_name=client_name, train_loader=train_loader, test_loader=test_loader,
                                        device=device))

    weights_0 = parameters2weights(server.model.parameters())
    weights_0_file = os.path.join(settings.WEIGHTS_DIR, 'weights_0')
    np.savez_compressed(weights_0_file, weights_0=weights_0[sampling_idx])
    cosines = {}
    for client_name in client_list:
        cosines[client_name] = []

    last_time = time()
    # Start federated learning
    for i in range(communication_rounds):
        print('Communication Round {} | Time: {}'.format(i, time() - last_time))
        last_time = time()

        global_parameters = server.get_parameters()
        local_parameters = []
        w0 = parameters2weights(server.model.parameters())
        # Federated Learning
        for client in federated_clients:
            client.set_parameters(global_parameters)
            client.run(n_epochs=epochs_per_round)
            local_parameters.append(client.get_parameters())

        server.aggregate(local_parameters)

        # Test
        for client in federated_clients:
            if client.name in client_list:
                client.test()

        if saving:
            server_weights = parameters2weights(server.model.parameters())
            weights_file = os.path.join(settings.WEIGHTS_DIR, 'Server_r{}'.format(i))
            np.savez_compressed(weights_file, server_weights=server_weights[sampling_idx])

            for client_id, client_name in enumerate(client_list):
                client = federated_clients[np.where(client_names == client_name)[0][0]]
                model = server.model
                model.eval()
                results = {}

                for sampling_type in sampling_types:
                    print('Predicting {} {}'.format(client_name, sampling_type))
                    data_loader = samples_data_loaders[sampling_type][client_id]
                    predictions = []

                    total = 0
                    correct = 0

                    with torch.no_grad():
                        for inputs, labels in data_loader:
                            inputs = inputs.float().to(device)
                            labels = labels.long().to(device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            predictions.append(predicted.cpu().numpy())
                            if sampling_type == 'local':
                                total += inputs.size(0)
                                correct += (predicted == labels).sum().item()
                    results[sampling_type] = np.concatenate(predictions)
                    if sampling_type == 'local':
                        print('Local Acc:', correct / total)

                output_file = os.path.join(settings.OUTPUTS_DIR, '{}_Server_r{}'.format(client_name, i))
                np.savez_compressed(output_file, **results)

                client_weights = parameters2weights(client.model.parameters())
                weights_file = os.path.join(settings.WEIGHTS_DIR, '{}_r{}'.format(client_name, i))
                np.savez_compressed(weights_file, client_weights=client_weights[sampling_idx])

                cosines[client_name].append(cos_v(client_weights - w0, server_weights - w0))

    loss_list = [client.history['loss'] for client in federated_clients if client.name in client_list]
    val_acc_list = [client.history['val_acc'] for client in federated_clients if client.name in client_list]
    np.savez_compressed(settings.VAL_FILE, client_names=client_names, loss=loss_list, val_acc=val_acc_list)

    cosines_file = os.path.join(settings.WEIGHTS_DIR, 'cosines')
    np.savez_compressed(cosines_file, **cosines)


if __name__ == '__main__':
    print('Running on device:', device)
    # local_learning(client_list=['Client-0', 'Client-1'], n_epochs=30)
    # local_predict()
    federated_learning(communication_rounds=200, epochs_per_round=1, saving=True, n_sampling_parameters=1000)

    # local_learning(client_names=['Client-0', 'Client-1'], n_epochs=50)
    # local_predict(client_names=['Client-0', 'Client-1'], sampling_types=['local', 'stratified'])
    # np.random.seed(0)
    # saved_paras_idx = np.random.permutation(23714)[:1000]
    # data = np.load(os.path.join(settings.DATA_HOME['face'], 'samples.npz'), allow_pickle=True)
    # federated_learning(client_names=['Client-0', 'Client-1'], communication_rounds=50, epochs_per_round=1,
    #                    samples_data=data, sampling_types=['local', 'stratified'], saving=True,
    #                    saved_paras_idx=saved_paras_idx)
