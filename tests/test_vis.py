import json

from tqdm import trange

from models.mnist import MnistModel
from datasets.mnist import get_mnist_data
from train import test
import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import settings
import os

s_name = 'Server'
c_name = 'Client-0'
ckpt_dir_client = os.path.join(settings.CHECKPOINT_DIR, c_name)
ckpt_dir_server = os.path.join(settings.CHECKPOINT_DIR, s_name)
history_file = os.path.join(settings.BASE_DIR, 'data', 'history_{}.json'.format(c_name))
rd = 500

if __name__ == '__main__':
    client_names, train_loaders, test_loaders, server_data_loader = get_mnist_data(batch_size=100)
    train_loader = train_loaders[np.array(client_names) == c_name][0]
    test_loader = test_loaders[np.array(client_names) == c_name][0]
    model = MnistModel().cuda()
    history = {
        'Server': {
            'loss': [],
            'val_acc': [],
        },
        'Client': {
            'loss': [],
            'val_acc': [],
        }
    }
    for r in range(rd):
        print('Round {}:'.format(r))
        model.load_state_dict(torch.load(os.path.join(ckpt_dir_server, '{}_r{}.cp'.format(s_name, r))))
        _, loss = test(train_loader, model, nn.CrossEntropyLoss(), 'cuda:0')
        acc, _ = test(test_loader, model, nn.CrossEntropyLoss(), 'cuda:0')
        print('Server | Train Loss: {}, Val_Acc: {}'.format(loss, acc))
        history['Server']['loss'].append(loss)
        history['Server']['val_acc'].append(acc)

        model.load_state_dict(torch.load(os.path.join(ckpt_dir_client, '{}_r{}.cp'.format(c_name, r))))
        _, loss = test(train_loader, model, nn.CrossEntropyLoss(), 'cuda:0')
        acc, _ = test(test_loader, model, nn.CrossEntropyLoss(), 'cuda:0')
        print('Client | Train Loss: {}, Val_Acc: {}'.format(loss, acc))
        history['Client']['loss'].append(loss)
        history['Client']['val_acc'].append(acc)

    with open(os.path.join(settings.BASE_DIR, 'data', 'history_{}.json'.format(c_name)), 'w') as f:
        json.dump(history, f)
    # loss = []
    # val_acc = []
    #
    # for r in trange(rd):
    #     ckpt_file = os.path.join(ckpt_dir, '{}_r{}.cp'.format(c_name, r))
    #     model.load_state_dict(torch.load(ckpt_file))
    #     _, _loss = test(test_loader=train_loader, model=model, criterion=nn.CrossEntropyLoss(), device='cuda:0')
    #     _acc, _ = test(test_loader=test_loader, model=model, criterion=nn.CrossEntropyLoss(), device='cuda:0')
    #     loss.append(_loss)
    #     val_acc.append(_acc)
    #
    #
    #
    # x = np.arange(0, rd, 1)
    #
    # plt.subplot(1, 2, 1)
    # plt.title('Train Loss')
    # plt.plot(x, loss)
    # plt.subplot(1, 2, 2)
    # plt.title('Validate Accuracy')
    # plt.plot(x, val_acc)
    # plt.show()
