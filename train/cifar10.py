import os
from time import time

import numpy as np
import torch

import settings
from datasets.cifar10 import get_cifar_data_loader, get_data_loader_by_samples
from models.cifar10 import Client, CIFARModel, Server
from utils import parameters2weights, cos_v

batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def local_learning(client_list, n_epochs):
    train_loaders = []
    test_loaders = []
    for client_name in client_list:
        train_loader, test_loader = get_cifar_data_loader(client_name, batch_size=batch_size, split=True)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    local_clients = []

    for client_name, train_loader, test_loader in zip(client_list, train_loaders, test_loaders):
        local_clients.append(Client(client_name=client_name + '_Local',
                                    train_loader=train_loader, test_loader=test_loader,
                                    checkpoint_path=os.path.join(settings.CHECKPOINTS_DIR, client_name),
                                    device=device))

    print('n_paras: {}'.format(sum(p.numel() for p in local_clients[0].model.parameters())))
    for client in local_clients:
        print('Training:', client.name)
        for epoch in range(n_epochs):
            client.run(n_epochs=1, save_last=True)
            client.test(save_best=True)


def local_predict(client_names):
    samples_data = np.load(os.path.join(settings.DATA_HOME['cifar10'], 'samples.npz'), allow_pickle=True)
    sampling_types = samples_data['sampling_types']
    client_list = samples_data['client_names']

    samples_data_loaders = {}
    for sampling_type in sampling_types:
        samples_data_loaders[sampling_type] = []
        for client_idx in range(client_list.shape[0]):
            data = samples_data[sampling_type][client_idx]
            labels = samples_data['ground_truth'][client_idx]
            data_loader = get_data_loader_by_samples(data, labels)
            samples_data_loaders[sampling_type].append(data_loader)

    for client_idx, client_name in enumerate(samples_data['client_names']):
        if client_name not in client_names:
            continue
        checkpoint_path = os.path.join(settings.CHECKPOINTS_DIR, client_name, '{}_Local_best.cp'.format(client_name))
        model = CIFARModel().to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        results = {}
        for sampling_type in samples_data['sampling_types']:
            print('Predicting {} {}'.format(client_name, sampling_type))
            # input_data = samples_data[sampling_type][client_idx]
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


def federated_learning(client_names, communication_rounds=1, epochs_per_round=1, saving=False,
                       n_sampling_parameters=1000):
    samples_data = np.load(os.path.join(settings.DATA_HOME['cifar10'], 'samples.npz'), allow_pickle=True)
    sampling_types = samples_data['sampling_types']
    client_list = samples_data['client_names']

    samples_data_loaders = {}
    for sampling_type in sampling_types:
        samples_data_loaders[sampling_type] = []
        for client_idx in range(client_list.shape[0]):
            data = samples_data[sampling_type][client_idx]
            labels = samples_data['ground_truth'][client_idx]
            data_loader = get_data_loader_by_samples(data, labels)
            samples_data_loaders[sampling_type].append(data_loader)

    train_loaders = []
    test_loaders = []
    for client_name in client_names:
        train_loader, test_loader = get_cifar_data_loader(client_name, batch_size=batch_size, split=True)
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
            client.test()

        if saving:
            server_weights = parameters2weights(server.model.parameters())
            weights_file = os.path.join(settings.WEIGHTS_DIR, 'Server_r{}'.format(i))
            np.savez_compressed(weights_file, server_weights=server_weights[sampling_idx])

            for client in federated_clients:
                client_name = client.name
                if client_name not in client_list:
                    continue
                client_id = np.where(client_name == client_list)[0][0]

                model = client.model
                results = {}

                for sampling_type in sampling_types:
                    print('Predicting {} {}'.format(client_name, sampling_type))
                    # input_data = samples_data[sampling_type][client_id]
                    data_loader = samples_data_loaders[sampling_type][client_id]
                    predictions = []
                    with torch.no_grad():
                        for inputs, labels in data_loader:
                            inputs = inputs.float().to(device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            predictions.append(predicted.cpu().numpy())
                    results[sampling_type] = np.concatenate(predictions)

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
    # local_learning(client_list=['Client-2', 'Client-7'], n_epochs=60)
    local_predict(['Client-2', 'Client-7'])
    # client_names = ['Client-{}'.format(i) for i in range(10)]
    # client_names = ['Client-0', 'Client-2', 'Client-7']
    # federated_learning(client_names=client_names, communication_rounds=50, epochs_per_round=1, saving=True,
    #                    n_sampling_parameters=1000)
