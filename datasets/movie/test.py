import os
from collections import Counter
from time import time

import numpy as np
import torch
import torchtext
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.datasets import IMDB, AmazonReviewPolarity
from torchtext.vocab import Vocab, GloVe

import settings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vec = GloVe(name='6B', dim=300, cache=os.path.join(settings.DATA_HOME['movie'], '.vector_cache'))

tokens = ['chip', 'baby', 'Beautiful']
ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
print(len(vec.itos))

# text_pipeline = lambda x: [vocab[token] + 1 for token in tokenizer(x)]
#
#     def pad_text(text, seq_len=200):
#         text_len = len(text)
#         if text_len > seq_len:
#             return np.array(text[:seq_len])
#         else:
#             return np.array([0] * (seq_len - text_len) + text)

# imdb_train_iter, imdb_test_iter = IMDB(root=settings.DATA_HOME['movie'])
#
# for label, line in imdb_train_iter:
#     ret = vec.get_vecs_by_tokens()
