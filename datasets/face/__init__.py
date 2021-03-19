import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

import settings

data_home = settings.DATA_HOME['face']


class FaceDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder

    def __getitem__(self, idx):
        image, target = self.image_folder[idx]
        return image, target

    def __len__(self):
        return len(self.image_folder)


class FaceSampleDataset(Dataset):
    def __init__(self, data, labels, transform):
        super(FaceSampleDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.labels[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return self.data.shape[0]


train_transforms = T.Compose([T.RandomResizedCrop((28, 28)),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize((0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5))
                              ])
val_transforms = T.Compose([T.ToTensor(),
                            T.Resize((28, 28)),
                            T.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))
                            ])

transforms = {
    'Client-0': {'train': T.Compose([T.RandomResizedCrop(28),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(),
                                     T.Normalize((0.56858784, 0.46486232, 0.41624302),
                                                 (0.27096274, 0.24747865, 0.24957362))
                                     ]),
                 'val': T.Compose([T.Resize(28),
                                   T.ToTensor(),
                                   T.Normalize((0.56858784, 0.46486232, 0.41624302),
                                               (0.27096274, 0.24747865, 0.24957362))
                                   ]),
                 },
    'Client-1': {'train': T.Compose([T.RandomResizedCrop((28, 28)),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(),
                                     T.Normalize((0.58048403, 0.51398385, 0.48651683),
                                                 (0.2924182, 0.28392276, 0.28706816))
                                     ]),
                 'val': T.Compose([T.Resize((28, 28)),
                                   T.ToTensor(),
                                   T.Normalize((0.58048403, 0.51398385, 0.48651683),
                                               (0.2924182, 0.28392276, 0.28706816))
                                   ]),
                 },
}


def get_face_data_loader(client_name, batch_size):
    train_data_path = os.path.join(data_home, client_name, 'Train')
    test_data_path = os.path.join(data_home, client_name, 'Test')
    val_data_path = os.path.join(data_home, client_name, 'Validate')

    # train_image_folder = ImageFolder(root=train_data_path, transform=transforms[client_name]['train'])
    # test_image_folder = ImageFolder(root=test_data_path, transform=transforms[client_name]['val'])
    # val_image_folder = ImageFolder(root=val_data_path, transform=transforms[client_name]['val'])

    train_image_folder = ImageFolder(root=train_data_path, transform=train_transforms)
    test_image_folder = ImageFolder(root=test_data_path, transform=val_transforms)
    val_image_folder = ImageFolder(root=val_data_path, transform=val_transforms)

    test_dataset = FaceDataset(test_image_folder)
    val_dataset = FaceDataset(val_image_folder)

    train_data_loader = DataLoader(dataset=FaceDataset(train_image_folder), batch_size=batch_size, shuffle=True,
                                   num_workers=4)
    test_data_loader = DataLoader(dataset=ConcatDataset([test_dataset, val_dataset]), batch_size=batch_size,
                                  shuffle=True, num_workers=4)

    return train_data_loader, test_data_loader


def get_data_loader_by_client(client_name, batch_size):
    train_data_path = os.path.join(data_home, client_name, 'Train')
    test_data_path = os.path.join(data_home, client_name, 'Test')
    val_data_path = os.path.join(data_home, client_name, 'Validate')

    # transform = T.Compose([T.Resize((28, 28)), T.ToTensor()])
    transform = val_transforms
    train_image_folder = ImageFolder(root=train_data_path, transform=transform)
    test_image_folder = ImageFolder(root=test_data_path, transform=transform)
    val_image_folder = ImageFolder(root=val_data_path, transform=transform)

    train_dataset = FaceDataset(train_image_folder)
    test_dataset = FaceDataset(test_image_folder)
    val_dataset = FaceDataset(val_image_folder)

    dataset = ConcatDataset([train_dataset, test_dataset, val_dataset])
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return data_loader


def new_data_loader(data, labels, batch_size=4):
    dateset = FaceSampleDataset(data, labels, transform=val_transforms)
    return DataLoader(dataset=dateset, batch_size=batch_size, shuffle=False, num_workers=4)


def get_sample_data_loaders():
    samples_data = np.load(os.path.join(settings.DATA_HOME['face'], 'samples.npz'), allow_pickle=True)
    sampling_types = samples_data['sampling_types']
    client_list = samples_data['client_names']

    samples_data_loaders = {}
    for sampling_type in sampling_types:
        samples_data_loaders[sampling_type] = []
        for client_idx in range(client_list.shape[0]):
            data = samples_data[sampling_type][client_idx]
            data = data.reshape((-1, 3, 28, 28)).transpose((0, 2, 3, 1))
            labels = samples_data['ground_truth'][client_idx]
            data_loader = new_data_loader(data, labels)
            samples_data_loaders[sampling_type].append(data_loader)

    return client_list, sampling_types, samples_data_loaders


def get_data_by_client(client_name):
    train_data_path = os.path.join(data_home, client_name, 'Train')
    test_data_path = os.path.join(data_home, client_name, 'Test')
    val_data_path = os.path.join(data_home, client_name, 'Validate')

    # transform = T.Compose([T.Resize((28, 28)), T.ToTensor()])
    transform = T.Compose([T.Resize((28, 28)), T.ToTensor()])
    train_image_folder = ImageFolder(root=train_data_path, transform=transform)
    test_image_folder = ImageFolder(root=test_data_path, transform=transform)
    val_image_folder = ImageFolder(root=val_data_path, transform=transform)

    images = []
    labels = []
    for image, label in train_image_folder:
        images.append(image.numpy() * 255)
        labels.append(label)

    for image, label in test_image_folder:
        images.append(image.numpy() * 255)
        labels.append(label)

    for image, label in val_image_folder:
        images.append(image.numpy() * 255)
        labels.append(label)
    return np.array(images, np.uint8), np.array(labels, np.long)
#
#
# def create_dataset():
#     images_0, labels_0 = get_data_by_client('Client-0')
#     images_0 = images_0.reshape(images_0.shape[0], -1)
#     images_1, labels_1 = get_data_by_client('Client-1')
#     images_1 = images_1.reshape(images_1.shape[0], -1)
#     np.savez_compressed(os.path.join(settings.DATA_HOME['face'], 'face_dataset'),
#                         data=[images_0, images_1],
#                         labels=[labels_0, labels_1],
#                         client_names=['Client-0', 'Client-1'])


def get_data_for_sampling(client_names):
    all_data = []
    all_labels = []
    for client_name in client_names:
        data, labels = get_data_by_client(client_name)
        data = data.reshape((-1, 3 * 28 * 28))
        all_data.append(data)
        all_labels.append(labels)
    ret = {'client_names': np.array(client_names),
           'data_shape': [3, 28, 28],
           'label_names': ['With Mask', 'Without Mask'],
           'data': np.array(all_data),
           'labels': np.array(all_labels),
           'train_size': [6000, 6000],
           'test_size': [1792, 1191],
           }
    return ret
