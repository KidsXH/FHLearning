import os

import numpy as np
import torch
from torchtext.vocab import Vocab

import settings
from datasets.movie import get_vocab_counter, get_data_loader
from models.movie import SentimentAnalysisModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_dim = 1
embedding_dim = 512
hidden_dim = 256
n_layers = 2
batch_size = 50
if __name__ == '__main__':

    vocab = Vocab(get_vocab_counter(), min_freq=1)
    vocab_size = len(vocab) + 1

    net = SentimentAnalysisModel(vocab_size=vocab_size,
                                 embedding_dim=embedding_dim,
                                 hidden_dim=hidden_dim,
                                 n_layers=n_layers,
                                 output_dim=output_dim,
                                 pad_idx=0).to(device)

    num_parameters = 0
    for name, paras in net.named_parameters():
        if name != 'embedding.weight':
            num_parameters += paras.numel()
    print('Number of parameters:', num_parameters)

    lr = 0.001

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip = 5

    train_loader, test_loader = get_data_loader('amazon', batch_size)

    net.train()

    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size, device)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            inputs = inputs.long().to(device)
            labels = labels.long().to(device)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size, device)
                val_losses = []
                num_correct = 0
                num_inputs = 0
                net.eval()
                for inputs, labels in test_loader:
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs = inputs.long().to(device)
                    labels = labels.long().to(device)

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                    pred = torch.round(output).long()  # type: torch.Tensor
                    correct = (pred == labels).sum().item()
                    num_correct += np.sum(correct)
                    num_inputs += inputs.size(0)
                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)),
                      'Val Acc: {:.6f}'.format(num_correct / num_inputs),
                      )
