import os

import numpy as np
from sklearn.decomposition import PCA

import settings
from datasets.cifar10 import get_data_for_sampling as get_cifar10
from datasets.face import get_data_for_sampling as get_face
from datasets.mnist import get_data_for_sampling as get_mnist

get_data_for_sampling = {
    'mnist': get_mnist,
    'face': get_face,
    'cifar10': get_cifar10,
}


def sampling_dataset(dataset_name, client_names):
    dataset = get_data_for_sampling[dataset_name](client_names)
    all_data = dataset['data']
    labels = dataset['labels']
    label_names = dataset['label_names']
    data_shape = dataset['data_shape']
    train_size = dataset['train_size']
    test_size = dataset['test_size']

    local_data = all_data
    stratified_samples = []
    systematic_samples = []

    for data in all_data:
        sampled_data = sample_data(data, 5, 5, 'stratified')
        stratified_samples.append(sampled_data)
        sampled_data = sample_data(data, 5, 5, 'systematic')
        systematic_samples.append(sampled_data)

    stratified_samples = np.array(stratified_samples, dtype=np.uint8)
    systematic_samples = np.array(systematic_samples, dtype=np.uint8)

    saved_data = {'client_names': client_names,
                  'sampling_types': ['local', 'stratified', 'systematic'],
                  'label_names': label_names,
                  'type': 'image',
                  'shape': data_shape,
                  'local': local_data,
                  'stratified': stratified_samples,
                  'systematic': systematic_samples,
                  'ground_truth': labels,
                  'train_size': train_size,
                  'test_size': test_size,
                  }

    data_file = os.path.join(settings.DATA_HOME[dataset_name], 'samples')
    np.savez_compressed(data_file, **saved_data)


def sample_data(data, n_points, n_dimensions, sampling_type, clip=True):
    pca = PCA(n_components=n_dimensions)
    t_data = pca.fit_transform(data)

    space = get_space(data=t_data, n_points=n_points, n_dimensions=n_dimensions, sampling_type=sampling_type)
    new_data = pca.inverse_transform(space)  # type:np.ndarray

    if clip:
        new_data = new_data.clip(np.min(data), np.max(data))

    return new_data


def get_space(data, n_points, n_dimensions, sampling_type):
    subspace = [[]]

    for d in range(n_dimensions):
        new_space = []
        min_v = data[:, d].min()
        max_v = data[:, d].max()
        if sampling_type == 'systematic':
            values = np.linspace(min_v, max_v, n_points)
        else:
            a = data[:, d]
            q = np.linspace(0, 100, n_points)
            values = np.percentile(a, q, interpolation='lower')

        for value in values:
            for point in subspace:
                new_space.append(point + [value])

        subspace = new_space[:]

    return subspace
