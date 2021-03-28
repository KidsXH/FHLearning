import torch
from torch import nn
from torch.nn.functional import relu, softmax

from models.csBase import ClientBase, ServerBase


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = softmax(x, dim=1)
        return x


class Client(ClientBase):
    def __init__(self, client_name, train_loader, test_loader, start_epoch=0, checkpoint_path='./', device='cpu',
                 best_acc=0):
        super(Client, self).__init__(client_name, train_loader, test_loader,
                                     start_epoch=start_epoch, checkpoint_path=checkpoint_path, device=device)
        self.model = MnistModel().to(device)
        # self.criterion = FocalLoss(2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.002, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995, last_epoch=start_epoch - 1)
        self.best_acc = best_acc

    def run(self, n_epochs, save_last=False):
        train_loader = self.train_loader
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.device

        total_loss = 0.0
        total_acc = 0.0

        model.train()

        for epoch in range(n_epochs):
            total = 0
            correct = 0
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total += inputs.size(0)
                correct += torch.sum(pred == labels.data).item()
                running_loss += loss.item() * inputs.size(0)
            scheduler.step()
            epoch_loss = running_loss / total
            epoch_acc = correct / total

            print('{} Train | Epoch={:d} | Loss={:.4f} | Acc={:.4f}'.format(self.name, self.cur_epoch,
                                                                            epoch_loss, epoch_acc))

            self.cur_epoch += 1
            total_loss += epoch_loss
            total_acc += epoch_acc

        if save_last:
            self.save(suffix='_last')

        self.history['acc'].append(total_acc / n_epochs)
        self.history['loss'].append(total_loss / n_epochs)

    def test(self, save_best=False):
        test_loader = self.test_loader
        model = self.model
        criterion = self.criterion
        device = self.device

        total = 0
        correct = 0
        running_loss = 0.0

        model.eval()

        with torch.no_grad():
            for (inputs, labels) in test_loader:
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                total += inputs.size(0)
                correct += torch.sum(pred == labels.data).item()
                running_loss += loss.item() * inputs.size(0)

        total_loss = running_loss / total
        total_acc = correct / total

        print(
            '{} Test  | Epoch={:d} | Loss={:.4f} | Acc={:.4f}'.format(self.name, self.cur_epoch - 1,
                                                                      total_loss, total_acc))

        if self.best_acc < total_acc:
            print('Achieve best accuracy: {}'.format(total_acc))
            self.best_acc = total_acc
            if save_best:
                self.save(suffix='_best')

        self.history['val_acc'].append(total_acc)
        self.history['val_loss'].append(total_loss)


class Server(ServerBase):
    def __init__(self, start_round=0, checkpoint_path='./', device='cpu'):
        super(Server, self).__init__(start_round=start_round, checkpoint_path=checkpoint_path,
                                     device=device)
        self.model = MnistModel().to(device)
        self.criterion = nn.CrossEntropyLoss()
