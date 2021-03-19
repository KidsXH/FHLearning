import os

import numpy as np
import torch

import settings
from models.mnist import MnistModel
from matplotlib import pyplot as plt
from utils import normalize, weights2dict

data_file = os.path.join(settings.DATA_HOME['mnist'], 'samples.npz')


def predict(model):
    data = np.load(data_file)['data']
    data = normalize(data)
    predictions = []
    with torch.no_grad():
        for inputs in data:
            inputs = torch.Tensor([inputs]).float().cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu().numpy())
    predictions = np.concatenate(predictions)

    # show_prediction(data, predictions, 1240, 1250)
    return predictions


def show_prediction(data, prediction, start, end):
    for image, pred in zip(data[start:end], prediction[start:end]):
        plt.figure()
        plt.title('Predict: {}'.format(pred))
        plt.imshow(image.reshape((28, 28)), cmap='gray')
        plt.show()


def draw_image(image):
    idx_0 = image < 0
    idx_1 = image > 255
    image[idx_0] = 0
    image[idx_1] = 255
    print(idx_0.sum(), idx_1.sum())
    image = image.reshape((28, 28))
    plt.imshow(image, cmap='gray')
    plt.show()


def get_server_model(cm_round, device):
    weights_file = os.path.join(settings.WEIGHTS_DIR, 'Server_weights_r{}.npy'.format(cm_round))
    weights = np.load(weights_file)
    model = MnistModel().to(device)
    state_dict = weights2dict(weights=weights, ref_dict=model.state_dict(), device='cuda:0')
    model.load_state_dict(state_dict)
    return model


def get_client_model(client_name, device):
    ckpt_file = os.path.join(settings.HISTORY_DIR, 'checkpoints', client_name, 'local_model_e50.cp')
    model = MnistModel().to(device)
    model.load_state_dict(torch.load(ckpt_file))
    return model


if __name__ == '__main__':
    model_server = get_server_model(cm_round=199, device='cuda:0')
    output_server = predict(model_server)
    model_client = get_client_model(client_name='Client-0', device='cuda:0')
    output_client = predict(model_client)

    output_file = os.path.join(settings.BASE_DIR, 'data', 'output')
    np.savez_compressed(output_file, output_client=output_client, output_server=output_server)
