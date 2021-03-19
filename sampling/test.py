import json
import os

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

import settings
from sampling import sampling_dataset


def draw_image(image):
    # idx_0 = image < 0
    # idx_1 = image > 255
    # image[idx_0] = 0
    # image[idx_1] = 255
    # print(idx_0.sum(), idx_1.sum())
    image = image.reshape((28, 28))
    plt.imshow(image, cmap='gray')
    plt.show()

    # plt.imsave(os.path.join(settings.BASE_DIR, 'data', 'images', 'sample_{}.bmp').format(idx), image, cmap='gray')


def save_image(image, idx):
    image = image.reshape((28, 28))
    plt.imsave(os.path.join(settings.BASE_DIR, 'data', 'images', 'sample_{}.png').format(idx), image, cmap='gray')


def pca_figure(data, labels):
    pca = PCA(n_components=2)
    X = pca.fit_transform(data)

    plt.scatter(*X.T, c=labels, alpha=0.6)
    plt.show()


if __name__ == '__main__':
    clients = ['Client-{}'.format(i) for i in range(10)]
    sampling_dataset('cifar10', clients)
    f = np.load(os.path.join(settings.DATA_HOME['cifar10'], 'samples.npz'), allow_pickle=True)

    # samples_data = np.load(os.path.join(settings.DATA_HOME['face'], 'samples.npz'))
    # data = samples_data['stratified'][0]
    # pca_figure(data, [0] * data.shape[0])

    # sampling_dataset('mnist')
    # data_file = os.path.join(settings.DATA_HOME['mnist'], 'samples.npz')
    # samples = np.load(data_file)
    # print(samples['named_labels'])
    #
    # saved_data = {}
    #
    # for k in samples:
    #     saved_data[k] = samples[k].tolist()
    #     print(k, np.shape(samples[k]))
    #
    # with open(os.path.join(settings.DATA_HOME['mnist'], 'samples.json'), 'w') as f:
    #     json.dump(saved_data, f)



    # data = np.load(data_file)
    # print(data['labels'])
    # data = samples['stratified'][0]
    # image = samples['local'][0][5688]
    # print(samples['ground_truth'][0][5688])
    # draw_image(image)
    # plt.figure(figsize=(1920, 1080))

    # rd_idx = np.random.permutation(np.arange(0, 3125, 1))[:16]
    # rd_idx = [1894, 5649, 414, 3982, 5478, 5772, 5831]
    # for idx, img in zip(rd_idx, data[rd_idx]):
    #     plt.title('idx={}'.format(idx))
    #     draw_image(img)
    #     save_image(img, idx)

