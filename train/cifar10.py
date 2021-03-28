import os
from time import time

import numpy as np
import torch

from settings import BASE_DIR
from datasets.cifar10 import get_cifar_data_loader, get_sample_data_loaders
from models.cifar10 import CIFARModel, Client, Server
from utils import get_sample_idx, sample_weights, update_cosines

batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HISTORY_DIR = os.path.join(BASE_DIR, 'data', 'history', 'cifar10')

CHECKPOINTS_DIR = os.path.join(HISTORY_DIR, 'checkpoints')
WEIGHTS_DIR = os.path.join(HISTORY_DIR, 'weights')
OUTPUTS_DIR = os.path.join(HISTORY_DIR, 'outputs')
VAL_FILE = os.path.join(HISTORY_DIR, 'validation.npz')


def local_learning(client_list, n_epochs):
    train_loaders = []
    test_loaders = []
    for client_name in client_list:
        train_loader, test_loader = get_cifar_data_loader(client_name, batch_size=batch_size)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    local_clients = []

    for client_name, train_loader, test_loader in zip(client_list, train_loaders, test_loaders):
        local_clients.append(Client(client_name=client_name + '_Local',
                                    train_loader=train_loader, test_loader=test_loader,
                                    checkpoint_path=os.path.join(CHECKPOINTS_DIR, client_name),
                                    device=device))

    print('n_paras: {}'.format(sum(p.numel() for p in local_clients[0].model.parameters())))
    for client in local_clients:
        print('Training:', client.name)
        for epoch in range(n_epochs):
            client.run(n_epochs=1, save_last=True)
            client.test(save_best=True)


def local_predict():
    client_names, sampling_types, samples_data_loaders = get_sample_data_loaders()

    for client_idx, client_name in enumerate(client_names):
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, client_name, '{}_Local_last.cp'.format(client_name))
        model = CIFARModel().to(device)
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
        output_file = os.path.join(OUTPUTS_DIR, '{}_local'.format(client_name))
        np.savez_compressed(output_file, **results)


def federated_learning(communication_rounds=1, epochs_per_round=1, saving=False,
                       sampling_idx_layers=None, sampling_idx_all=None):
    client_list, sampling_types, samples_data_loaders = get_sample_data_loaders()
    client_names = np.array(['Client-{}'.format(i) for i in range(4)])
    train_loaders = []
    test_loaders = []
    for client_name in client_names:
        train_loader, test_loader = get_cifar_data_loader(client_name, batch_size=batch_size)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    # Initiate Parameters
    server = Server(start_round=0, checkpoint_path=os.path.join(CHECKPOINTS_DIR, 'Server'), device=device)

    n_paras = sum(p.numel() for p in server.model.parameters())
    print('Total n_paras: {}'.format(n_paras))

    layer_names = server.model.layer_names
    n_layers = len(layer_names)
    info_file = os.path.join(HISTORY_DIR, 'model_info')
    np.savez_compressed(info_file, layer_names=layer_names)

    federated_clients = []

    for client_name, train_loader, test_loader in zip(client_names, train_loaders, test_loaders):
        federated_clients.append(Client(client_name=client_name,
                                        checkpoint_path=os.path.join(CHECKPOINTS_DIR, client_name),
                                        train_loader=train_loader, test_loader=test_loader, device=device))

    weights0_layers, weights0_all = sample_weights(server.model, sampling_idx_layers, sampling_idx_all)
    weights0_file = os.path.join(WEIGHTS_DIR, 'weights_0')
    np.savez_compressed(weights0_file, layers=weights0_layers, all=weights0_all)
    torch.save(server.model.state_dict(), os.path.join(CHECKPOINTS_DIR, 'model_0.cp'))

    cosines = {}
    for client_name in client_list:
        cosines[client_name] = [[] for _ in range(n_layers + 1)]

    total_accuracy_list = [[] for _ in client_list]

    last_time = time()
    # Start federated learning
    for i in range(communication_rounds):
        print('Communication Round {} | Time: {}'.format(i, time() - last_time))
        last_time = time()

        pre_model = CIFARModel().cuda()
        pre_model.load_state_dict(server.model.state_dict())

        global_parameters = server.get_parameters()
        local_parameters = []
        # Federated Learning
        for client in federated_clients:
            client.set_parameters(global_parameters)
            client.run(n_epochs=epochs_per_round, save_last=True)
            local_parameters.append(client.get_parameters())

        server.aggregate(local_parameters)
        server.save(suffix='_r{}'.format(i))

        if saving:
            server_weights_layers, server_weights_all = sample_weights(server.model,
                                                                       sampling_idx_layers, sampling_idx_all)
            weights_file = os.path.join(WEIGHTS_DIR, 'Server_r{}'.format(i))
            np.savez_compressed(weights_file, layers=server_weights_layers, all=server_weights_all)

            for client_id, client_name in enumerate(client_list):
                client = federated_clients[np.where(client_names == client_name)[0][0]]
                model = server.model
                model.eval()
                results = {}

                for sampling_type in sampling_types:
                    print('Predicting {} {}'.format(client_name, sampling_type))
                    data_loader = samples_data_loaders[sampling_type][client_id]
                    predictions = []
                    total, correct = 0, 0

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
                        print('    Total Acc:', correct / total)
                        total_accuracy_list[client_id].append(correct / total)

                output_file = os.path.join(OUTPUTS_DIR, '{}_Server_r{}'.format(client_name, i))
                np.savez_compressed(output_file, **results)

                client_weights_layers, client_weights_all = sample_weights(client.model,
                                                                           sampling_idx_layers, sampling_idx_all)

                weights_file = os.path.join(WEIGHTS_DIR, '{}_r{}'.format(client_name, i))
                np.savez_compressed(weights_file, layers=client_weights_layers, all=client_weights_all)

                update_cosines(pre_model, client.model, server.model, cosines[client_name])

        # Test
        for client in federated_clients:
            if client.name in client_list:
                client.set_parameters(server.get_parameters())
                client.test()

    loss_list = [client.history['loss'] for client in federated_clients if client.name in client_list]
    val_acc_list = [client.history['val_acc'] for client in federated_clients if client.name in client_list]
    np.savez_compressed(VAL_FILE, client_names=client_names, loss=loss_list, val_acc=val_acc_list,
                        tot_acc=total_accuracy_list)

    cosines_file = os.path.join(WEIGHTS_DIR, 'cosines')
    np.savez_compressed(cosines_file, **cosines)
