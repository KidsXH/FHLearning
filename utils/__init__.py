import copy

import numpy as np
import torch


def train(train_loader, model, criterion, optimizer, scheduler, device):
    total = 0
    correct = 0
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)
        # reset grad
        optimizer.zero_grad()
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total += inputs.size(0)
        correct += torch.sum(pred == labels.data).float()
        running_loss += loss.item() * inputs.size(0)
    scheduler.step()
    return correct / total, running_loss / total


def test(test_loader, model, criterion, device):
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += inputs.size(0)
            correct += torch.sum(pred == labels.data).float()
            running_loss += loss.item() * inputs.size(0)
    return correct / total, running_loss / total


def parameters2weights(parameters):
    weights = np.concatenate([paras.data.cpu().numpy().reshape(-1) for paras in parameters])
    return weights


def dict2weights(parameters):
    weights = np.concatenate([paras.cpu().numpy().reshape(-1) for paras in parameters.values()])
    return weights


def normalize(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def weights2dict(weights, ref_dict, device):
    new_dict = copy.deepcopy(ref_dict)
    cur_idx = 0
    for key in new_dict:
        shape_paras = new_dict[key].shape
        n_paras = np.prod(shape_paras)
        weights_cut = np.reshape(weights[cur_idx: cur_idx + n_paras], shape_paras)
        new_dict[key] = torch.Tensor(weights_cut).to(device)

    return new_dict


def cos_v(v1: np.ndarray, v2: np.ndarray):
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def sample_weights_layers(layers, sample_idx):
    weights = []

    for layer, idx in zip(layers, sample_idx):
        weights.append(parameters2weights(layer.parameters())[idx])

    return weights


def sample_weights_all(model, sample_idx):
    return parameters2weights(model.parameters())[sample_idx]


def sample_weights(model, sampling_idx_layers, sampling_idx_all):
    weights_layers = sample_weights_layers(model.layers, sampling_idx_layers)
    weights_all = sample_weights_all(model, sampling_idx_all)
    return weights_layers, weights_all


def get_cosine_layers(pre_layers, client_layers, server_layers):
    cosine_layers = []
    for pre_layer, client_layer, server_layer in zip(pre_layers, client_layers, server_layers):
        w0 = parameters2weights(pre_layer.parameters())
        w1 = parameters2weights(client_layer.parameters())
        w2 = parameters2weights(server_layer.parameters())
        cosine_layers.append(cos_v(w1 - w0, w2 - w0))
    return cosine_layers


def get_cosine_all(pre_parameters, client_parameters, server_parameters):
    w0 = parameters2weights(pre_parameters)
    w1 = parameters2weights(client_parameters)
    w2 = parameters2weights(server_parameters)
    return cos_v(w1 - w0, w2 - w0)


def update_cosines(pre_model, client_model, server_model, cosines):
    cosine_layers = get_cosine_layers(pre_model.layers, client_model.layers, server_model.layers)
    cosine_all = get_cosine_all(pre_model.parameters(), client_model.parameters(), server_model.parameters())
    cosine_layers = cosine_layers + [cosine_all]
    for c_all, c in zip(cosines, cosine_layers):
        c_all.append(c)


def get_sample_idx(model, max_num_parameters):
    np.random.seed(0)
    layer_names = model.layer_names
    layers = model.layers
    total = 0

    sampling_idx_layers = []

    for layer_name, layer in zip(layer_names, layers):
        n_paras = np.sum([p.numel() for p in layer.parameters()])
        total += n_paras

        sampling_idx_layers.append(sample_n_idx(n_paras, max_num_parameters))

        print(layer_name, n_paras)

    print('Total', total)

    sampling_idx_all = sample_n_idx(total, max_num_parameters)

    return sampling_idx_layers, sampling_idx_all


def sample_n_idx(n, max_n):
    if n > max_n:
        return np.random.permutation(n)[:max_n]
    else:
        return np.arange(0, n, 1)
