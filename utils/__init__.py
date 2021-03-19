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
