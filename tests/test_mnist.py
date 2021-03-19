import json
import os
from time import time

import torch
import numpy as np

import settings
from datasets.mnist import get_mnist_data
from models.mnist import Server, Client, MnistModel

batch_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_home = settings.DATA_HOME['mnist']
train_file = os.path.join(data_home, 'train.json')
test_file = os.path.join(data_home, 'test.json')
history_file = os.path.join(settings.BASE_DIR, 'data', 'history_r50.json')
samples_file = os.path.join(settings.DATA_HOME['mnist'], 'samples.npz')


def local_learning(n_clients, epochs):
    client_names, train_loaders, test_loaders = get_mnist_data(batch_size=batch_size)
    local_clients = []

    for client_name, train_loader, test_loader in zip(client_names, train_loaders, test_loaders):
        local_clients.append(Client(client_name=client_name, train_loader=train_loader, test_loader=test_loader,
                                    checkpoint_path=os.path.join(settings.CHECKPOINTS_DIR, client_name),
                                    device=device))

    local_clients = [local_clients[0], local_clients[2]]
    # local_clients = local_clients[:n_clients]

    for client in local_clients:
        print(client.name)
        client.run(n_epochs=epochs)
        ckpt_file = os.path.join(settings.CHECKPOINTS_DIR, client.name, 'local_model_e{}.cp'.format(epochs))
        torch.save(client.model.state_dict(), ckpt_file)
        # client.model.load_state_dict(torch.load(ckpt_file))


def local_predict(data, sampling_types, next_epoch=50):
    client_names = data['client_names']
    client_list = ['Client-0', 'Client-2']

    for client_idx, client_name in enumerate(client_names):
        if client_name not in client_list:
            continue
        checkpoint_path = os.path.join(settings.CHECKPOINTS_DIR, client_name, 'local_model_e{}.cp'.format(next_epoch))
        model = MnistModel().to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        results = {}
        for sampling_type in sampling_types:
            print('Predicting {} {}'.format(client_name, sampling_type))
            input_data = data[sampling_type][client_idx]
            predictions = []
            with torch.no_grad():
                for inputs in input_data:
                    inputs = torch.Tensor([inputs]).float().cuda()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.append(predicted.cpu().numpy())

            results[sampling_type] = np.concatenate(predictions)
        output_file = os.path.join(settings.OUTPUTS_DIR, '{}_local'.format(client_name))
        np.savez_compressed(output_file, **results)


def federated_learning(n_clients=10, communication_rounds=1, epochs_per_round=1,
                       samples_data=None, sampling_types=None, saving=False, saved_paras_idx=None):
    client_names, train_loaders, test_loaders = get_mnist_data(batch_size=batch_size)

    # Initiate Parameters
    server = Server(start_round=0, checkpoint_path=os.path.join(settings.CHECKPOINTS_DIR, 'Server'), device=device)

    # print('n_paras: {}'.format(sum(p.numel() for p in server.model.parameters())))

    federated_clients = []

    for client_name, train_loader, test_loader in zip(client_names, train_loaders, test_loaders):
        federated_clients.append(Client(client_name=client_name, train_loader=train_loader, test_loader=test_loader,
                                        device=device))

    federated_clients = federated_clients[:n_clients]
    client_names = client_names[:n_clients]

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
            print('{} training | '.format(client.name), end='')
            client.set_parameters(global_parameters)
            client.run(n_epochs=epochs_per_round)
            local_parameters.append(client.get_parameters())

        server.aggregate(local_parameters)

        # Evaluate
        for client in federated_clients:
            print('{} evaluating | '.format(client.name), end='')
            client.evaluate()

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
                            inputs = torch.Tensor([inputs]).float().cuda()
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


def cos_v(v1: np.ndarray, v2: np.ndarray):
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def predict(model, input_data):
    # with torch.no_grad():
    #     for (inputs, labels) in data_loader:
    #         inputs = inputs.float().to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         predictions.append(predicted.cpu().numpy())
    predictions = []
    with torch.no_grad():
        for inputs in input_data:
            inputs = torch.Tensor([inputs]).float().cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu().numpy())
    predictions = np.concatenate(predictions)
    return predictions


def parameters2weights(parameters):
    weights = np.concatenate([paras.data.cpu().numpy().reshape(-1) for paras in parameters])
    return weights


if __name__ == '__main__':
    predict_data = np.load(os.path.join(settings.DATA_HOME['mnist'], 'samples.npz'))
    # local_learning(n_clients=10, epochs=50)
    # local_predict(predict_data, sampling_types=['local', 'stratified'])
    # d = np.load(os.path.join(settings.OUTPUTS_DIR, '{}_outputs_client.npz'.format('Client-0')))
    # print(d['outputs_client'].shape)

    # ----------------------------------Federated Training------------------------------------
    np.random.seed(0)
    saved_paras_idx = np.random.permutation(203530)[:1000]
    federated_learning(n_clients=10, communication_rounds=1, epochs_per_round=1,
                       samples_data=predict_data, sampling_types=['local', 'stratified'], saving=False,
                       saved_paras_idx=saved_paras_idx)
