import copy
import os

import torch

from utils import parameters2weights, dict2weights, train, test


class ClientBase:
    def __init__(self, client_name, train_loader, test_loader, start_epoch,
                 checkpoint_path, device):
        self.name = client_name
        self.device = device
        self.model = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.history = {'clientName': client_name, 'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
        self.cur_epoch = start_epoch
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

    def get_client_name(self):
        return self.name

    def get_parameters(self):
        return self.model.state_dict()

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

    def save(self, suffix=''):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, self.name + suffix + '.cp'))

    def run(self, n_epochs):
        loss = 0.0
        acc = 0.0
        for epoch in range(n_epochs):
            acc, loss = train(train_loader=self.train_loader, model=self.model, criterion=self.criterion,
                              optimizer=self.optimizer, scheduler=self.scheduler, device=self.device)

            print('Epoch {}: Accuracy - {}, Loss - {}'.format(self.cur_epoch, acc, loss))
            self.cur_epoch += 1

        self.history['acc'].append(acc)
        self.history['loss'].append(loss)

    def evaluate(self):
        val_acc, val_loss = test(test_loader=self.test_loader, model=self.model, criterion=self.criterion,
                                 device=self.device)
        print('Accuracy: {} Validate Loss: {}'.format(val_acc, val_loss))

        self.history['val_acc'].append(val_acc)
        self.history['val_loss'].append(val_loss)


class ServerBase:
    def __init__(self, start_round, checkpoint_path, device):
        self.model = None
        self.device = device
        self.cur_round = start_round
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

    def get_parameters(self):
        return self.model.state_dict()

    def aggregate(self, local_parameters):
        """Aggregate parameters using FedAvg
        """
        clients_number = len(local_parameters)
        aggregated_parameters = copy.deepcopy(local_parameters[0])

        for parameters in local_parameters[1:]:
            _a = copy.deepcopy(aggregated_parameters)
            for item in parameters:
                aggregated_parameters[item] += parameters[item]

        for item in aggregated_parameters:
            aggregated_parameters[item] /= clients_number

        self.model.load_state_dict(aggregated_parameters)

    def save(self, suffix):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, 'Server' + suffix + '.cp'))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
