import os
from time import time

import numpy as np
import torch

import settings
from datasets.anime import get_anime_data_loaders
from models.anime import Client, AnimeModel, Server
from utils import parameters2weights, cos_v

batch_size = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def local_learning(client_list, n_epochs):
    client_names, train_loaders, test_loaders = get_anime_data_loaders(batch_size=batch_size)

    local_clients = []

    for client_name, train_loader, test_loader in zip(client_names, train_loaders, test_loaders):
        if client_name not in client_list:
            continue
        local_clients.append(Client(client_name=client_name + '_Local',
                                    train_loader=train_loader, test_loader=test_loader,
                                    checkpoint_path=os.path.join(settings.CHECKPOINTS_DIR, client_name),
                                    device=device))

    for client in local_clients:
        print('Training:', client.name)
        for epoch in range(n_epochs):
            client.run(n_epochs=1, save_last=True)
            client.test(save_best=True)


def local_predict(client_names, sampling_types):
    samples_data = np.load(os.path.join(settings.DATA_HOME['anime'], 'samples.npz'), allow_pickle=True)

    for client_idx, client_name in enumerate(samples_data['client_names']):
        if client_name not in client_names:
            continue
        checkpoint_path = os.path.join(settings.CHECKPOINTS_DIR, client_name, '{}_Local_last.cp'.format(client_name))
        model = AnimeModel().to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        results = {}
        for sampling_type in sampling_types:
            print('Predicting {} {}'.format(client_name, sampling_type))
            input_data = samples_data[sampling_type][client_idx]
            predictions = []
            with torch.no_grad():
                for inputs in input_data:
                    inputs = torch.tensor([inputs]).float().to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.append(predicted.cpu().numpy())

            results[sampling_type] = np.concatenate(predictions)
        output_file = os.path.join(settings.OUTPUTS_DIR, '{}_local'.format(client_name))
        np.savez_compressed(output_file, **results)


def federated_learning(client_list, communication_rounds=1, epochs_per_round=1,
                       samples_data=None, sampling_types=None, saving=False, saved_paras_idx=None):
    client_names, train_loaders, test_loaders = get_anime_data_loaders(batch_size=batch_size)

    # Initiate Parameters
    server = Server(start_round=0, checkpoint_path=os.path.join(settings.CHECKPOINTS_DIR, 'Server'), device=device)

    print('n_paras: {}'.format(sum(p.numel() for p in server.model.parameters())))

    federated_clients = []

    for client_name, train_loader, test_loader in zip(client_names, train_loaders, test_loaders):
        if client_name not in client_list:
            continue
        federated_clients.append(Client(client_name=client_name, train_loader=train_loader, test_loader=test_loader,
                                        device=device))

    weights_0 = parameters2weights(server.model.parameters())
    weights_0_file = os.path.join(settings.WEIGHTS_DIR, 'weights_0')
    np.savez_compressed(weights_0_file, weights_0=weights_0[saved_paras_idx])
    cosines = {}
    for client_name in client_names:
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
            np.savez_compressed(weights_file, server_weights=server_weights[saved_paras_idx])

            for cid, client in enumerate(federated_clients):
                client_name = client_names[cid]
                model = client.model
                results = {}

                for sampling_type in sampling_types:
                    print('Predicting {} {}'.format(client_name, sampling_type))
                    input_data = samples_data[sampling_type][cid]
                    predictions = []
                    with torch.no_grad():
                        for inputs in input_data:
                            inputs = torch.tensor([inputs]).float().to(device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            predictions.append(predicted.cpu().numpy())
                    results[sampling_type] = np.concatenate(predictions)

                output_file = os.path.join(settings.OUTPUTS_DIR, '{}_Server_r{}'.format(client_name, i))
                np.savez_compressed(output_file, **results)

                client_weights = parameters2weights(client.model.parameters())
                weights_file = os.path.join(settings.WEIGHTS_DIR, '{}_r{}'.format(client_name, i))
                np.savez_compressed(weights_file, client_weights=client_weights[saved_paras_idx])

                cosines[client_name].append(cos_v(client_weights - w0, server_weights - w0))

    loss_list = [client.history['loss'] for client in federated_clients]
    val_acc_list = [client.history['val_acc'] for client in federated_clients]
    np.savez_compressed(settings.VAL_FILE, client_names=client_names, loss=loss_list, val_acc=val_acc_list)

    cosines_file = os.path.join(settings.WEIGHTS_DIR, 'cosines')
    np.savez_compressed(cosines_file, **cosines)


if __name__ == '__main__':
    print('Running on device:', device)
    local_learning(client_list=['Poland'], n_epochs=50)
    # local_predict(client_names=['Client-0', 'Client-1'], sampling_types=['local', 'stratified'])
    # np.random.seed(0)
    # saved_paras_idx = np.random.permutation(23714)[:1000]
    # data = np.load(os.path.join(settings.DATA_HOME['face'], 'samples.npz'), allow_pickle=True)
    # federated_learning(client_names=['Client-0', 'Client-1'], communication_rounds=50, epochs_per_round=1,
    #                    samples_data=data, sampling_types=['local', 'stratified'], saving=True,
    #                    saved_paras_idx=saved_paras_idx)
