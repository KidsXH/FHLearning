import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

import settings
from datasets.face import get_face_data_loader, create_dataset

root = settings.DATA_HOME['face']


# Client-0: (0.56858784, 0.46486232, 0.41624302), (0.27096274, 0.24747865, 0.24957362)
# Client-1: (0.58048403, 0.51398385, 0.48651683), (0.2924182, 0.28392276, 0.28706816)
def calculate_mean_std(client_name):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

    data_path = os.path.join(root, client_name, 'Train')
    data_all = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    images = []
    for data, target in data_all:
        images.append(data.numpy())
    images = np.array(images)

    mean = (np.mean(images[:, 0, :, :]), np.mean(images[:, 1, :, :]), np.mean(images[:, 2, :, :]))
    std = (np.std(images[:, 0, :, :]), np.std(images[:, 1, :, :]), np.std(images[:, 2, :, :]))
    return mean, std


if __name__ == '__main__':
    create_dataset()
    # print(calculate_mean_std('Client-1'))
    # transform = transforms.Compose([transforms.Resize((28, 28)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.58048403, 0.51398385, 0.48651683),
    #                                                      (0.2924182, 0.28392276, 0.28706816))
    #                                 ])
    # test_data_path = os.path.join(settings.DATA_HOME['face'], 'Client-1', 'Validate')
    # test_image_folder = ImageFolder(root=test_data_path, transform=transform)
    #
    # depth = []
    # for idx, (img_file, label) in enumerate(test_image_folder.imgs):
    #     image = Image.open(img_file)
    #     if np.shape(image)[-1] > 3:
    #         print(idx, np.shape(image), img_file)
