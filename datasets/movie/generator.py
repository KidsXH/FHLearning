import os
from collections import Counter
from string import punctuation
from time import time

import numpy as np
from torchtext.data import get_tokenizer
from torchtext.datasets import AmazonReviewPolarity, IMDB
from torchtext.vocab import Vocab

import settings
from datasets.movie import get_vocab_counter

data_home = settings.DATA_HOME['movie']
amazon_file = os.path.join(data_home, 'amazon_reviews.npz')
imdb_file = os.path.join(data_home, 'imdb_reviews.npz')
dictionary_file = os.path.join(data_home, 'dictionary.npy')


def preprocess_data(pretrained_vocab=False):
    last_time = time()
    vocab = None
    tokenizer = get_tokenizer('basic_english')

    if pretrained_vocab:
        vocab = Vocab(get_vocab_counter())
    else:
        amazon_train_iter = AmazonReviewPolarity(root=settings.DATA_HOME['movie'], split='train')
        imdb_train_iter, imdb_test_iter = IMDB(root=settings.DATA_HOME['movie'])

        print('Data Loaded', time() - last_time)
        last_time = time()

        counter = Counter()

        for label, line in imdb_train_iter:
            counter.update(tokenizer(line))

        for label, line in imdb_test_iter:
            counter.update(tokenizer(line))

        print('IMDB Countered', time() - last_time)
        last_time = time()

        neg_count = 0
        for label, line in amazon_train_iter:
            if label == 1:
                neg_count += 1
                counter.update(tokenizer(line))
            if neg_count == 25000:
                break

        pos_count = 0
        for label, line in amazon_train_iter:
            if label == 2:
                pos_count += 1
                counter.update(tokenizer(line))
            if pos_count == 25000:
                break

        print('Amazon Countered', time() - last_time)
        last_time = time()

        vocab = Vocab(counter, min_freq=1)
        np.save(os.path.join(settings.DATA_HOME['movie'], 'vocab_counter'), counter)

    print('Vocab({}) Built'.format(len(vocab.freqs)), time() - last_time)
    last_time = time()

    text_pipeline = lambda x: [vocab[token] + 1 for token in tokenizer(x)]

    def pad_text(text, seq_len=200):
        text_len = len(text)
        if text_len > seq_len:
            return np.array(text[:seq_len])
        else:
            return np.array([0] * (seq_len - text_len) + text)

    amazon_train_iter = AmazonReviewPolarity(root=settings.DATA_HOME['movie'], split='train')
    imdb_train_iter, imdb_test_iter = IMDB(root=settings.DATA_HOME['movie'])
    print('Data Loaded', time() - last_time)
    last_time = time()

    imdb_dataset = []
    for label, line in imdb_train_iter:
        imdb_dataset.append([0 if label == 'neg' else 1, pad_text(text_pipeline(line))])

    for label, line in imdb_test_iter:
        imdb_dataset.append([0 if label == 'neg' else 1, pad_text(text_pipeline(line))])

    print('IMDB Transformed', time() - last_time)
    last_time = time()

    amazon_dataset = []
    neg_count = 0
    for label, line in amazon_train_iter:
        if label == 1:
            neg_count += 1
            amazon_dataset.append([label - 1, pad_text(text_pipeline(line))])
        if neg_count == 25000:
            break

    pos_count = 0
    for label, line in amazon_train_iter:
        if label == 2:
            pos_count += 1
            amazon_dataset.append([label - 1, pad_text(text_pipeline(line))])
        if pos_count == 25000:
            break

    print('Amazon Transformed', time() - last_time)
    last_time = time()

    np.save(os.path.join(settings.DATA_HOME['movie'], 'amazon'), amazon_dataset)
    np.save(os.path.join(settings.DATA_HOME['movie'], 'imdb'), imdb_dataset)

    print('Saved', time() - last_time)


def generate_dict(reviews, dictionary=None):
    words = reviews.reshape(-1)
    if dictionary is None:
        dictionary = {}
    for word in words:
        if word != '' and word not in dictionary:
            dictionary[word] = len(dictionary) + 1
    return dictionary


def split_words(reviews, max_length=200):
    new_reviews = []
    for review in reviews:  # type:str
        review = review.lower()
        review = ''.join([ch for ch in review if ch not in punctuation])
        words = review.split()

        if len(words) > max_length:
            words = words[:max_length]
        else:
            words = words + [''] * (max_length - len(words))
        new_reviews.append(words)
    return np.array(new_reviews)


if __name__ == '__main__':
    preprocess_data(pretrained_vocab=True)
