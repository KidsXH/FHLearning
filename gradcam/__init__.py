import os

import numpy as np
import torch
from torch.nn.functional import relu, interpolate
from matplotlib import pyplot as plt

from settings import BASE_DIR
from models.cifar10 import CIFARModel
from datasets.cifar10 import get_cifar_data_loader

HISTORY_DIR = os.path.join(BASE_DIR, 'data', 'history', 'cifar10')

CHECKPOINTS_DIR = os.path.join(HISTORY_DIR, 'checkpoints')
WEIGHTS_DIR = os.path.join(HISTORY_DIR, 'weights')
OUTPUTS_DIR = os.path.join(HISTORY_DIR, 'outputs')
VAL_FILE = os.path.join(HISTORY_DIR, 'validation.npz')


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def grad_cam(model, image, label, output_layers):
    image = image.unsqueeze(0).cuda()  # type: torch.Tensor
    image.requires_grad = True
    if image.grad is not None:
        image.grad.data.zero_()
    target = torch.tensor([label]).long().cuda()
    layers = model.layers

    features = []
    inputs = image
    outputs = None

    for idx, layer in enumerate(layers):
        outputs = layer(inputs)
        if idx in output_layers:
            features.append(outputs)
        inputs = outputs

    for feature in features:  # type: torch.Tensor
        feature.retain_grad()

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, target)  # type: torch.Tensor
    loss.backward()

    thermos = []

    for feature in features:
        thermo_grad = feature.grad.detach()
        thermo = -1 * thermo_grad.detach() * feature.detach()
        thermo = torch.sum(relu(thermo, inplace=True), dim=1, keepdim=True)
        thermo = interpolate(thermo, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=True)
        feature.grad.data.zero_()
        thermo = thermo.squeeze()
        thermo = thermo / (torch.max(thermo) + 1e-9)
        thermos.append(thermo.cpu())

    image_grad = relu(-1 * image.grad, inplace=True).detach()
    image_grad = image_grad.squeeze()
    image_grad = image_grad.permute([1, 2, 0]) / (torch.max(image_grad) + 1e-9)

    return thermos, image_grad.cpu()


if __name__ == '__main__':
    ckpt = os.path.join(CHECKPOINTS_DIR, 'Client-0', 'Client-0_Local_last.cp')

    cifar_model = CIFARModel().cuda()
    cifar_model.load_state_dict(torch.load(ckpt))

    train_loader, test_loader = get_cifar_data_loader('Client-0', 4)

    image_0, label_0 = train_loader.dataset[0]

    thermos, image_grad = grad_cam(cifar_model, image_0, label_0, [0, 1])

    plt.title('Raw Image')
    plt.imshow(image_0.permute([1, 2, 0]) / 2 + 0.5)
    plt.show()

    for idx, thermo in enumerate(thermos):
        plt.title('Thermo {}'.format(idx))
        plt.imshow(thermo)
        plt.show()

    plt.title('Image Grad')
    plt.imshow(image_grad)
    plt.show()
