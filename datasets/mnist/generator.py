from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import settings
import os


data_file = os.path.join(settings.DATA_HOME['mnist'], 'mnist_dataset')

groups_per_class = 2
samples_per_group = 3000
random_seed = 123

# Get MNIST data, normalize, and divide by level
mnist = fetch_openml("mnist_784", data_home=settings.DATA_HOME['mnist'])
mnist_data = []

for i in range(10):
    idx = mnist.target == str(i)
    mnist_data.append(mnist.data[idx])
    mnist_data[-1] = mnist_data[-1][:samples_per_group * groups_per_class]

data = []
target = []

seq_a = np.arange(0, 10, 1)
seq_b = np.concatenate((np.arange(1, 10, 1), [0]))

for a, b in zip(seq_a, seq_b):
    x = np.concatenate((mnist_data[a][:samples_per_group], mnist_data[b][samples_per_group:]))
    y = np.array([a] * samples_per_group + [b] * samples_per_group, dtype=int)
    data.append(x)
    target.append(y)

syn_data = {'client_names': [], 'label_distribution': [],
            'data': [], 'labels': []}

for uid in trange(10):
    username = 'Client-{}'.format(uid)
    x = data[uid]
    y = target[uid]

    syn_data['client_names'].append(username)
    syn_data['label_distribution'].append((seq_a[uid], seq_b[uid]))

    syn_data['data'].append(x)
    syn_data['labels'].append(y)

syn_data['data'] = np.array(syn_data['data'], np.uint8)
syn_data['labels'] = np.array(syn_data['labels'], np.long)

for (k, v) in syn_data.items():
    print('{}: {}'.format(k, np.shape(v)))

np.savez_compressed(data_file, **syn_data)
