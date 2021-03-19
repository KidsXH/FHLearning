import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

DATA_HOME = {
    'mnist': os.path.join(BASE_DIR, 'data', 'mnist'),
    'movie': os.path.join(BASE_DIR, 'data', 'movie'),
    'face': os.path.join(BASE_DIR, 'data', 'face'),
    'anime': os.path.join(BASE_DIR, 'data', 'anime'),
    'nutrition': os.path.join(BASE_DIR, 'data', 'nutrition'),
    'cifar10': os.path.join(BASE_DIR, 'data', 'cifar10'),
}


HISTORY_DIR = os.path.join(BASE_DIR, 'data', 'history', 'mnist')

CHECKPOINTS_DIR = os.path.join(HISTORY_DIR, 'checkpoints')
WEIGHTS_DIR = os.path.join(HISTORY_DIR, 'weights')
OUTPUTS_DIR = os.path.join(HISTORY_DIR, 'outputs')
VAL_FILE = os.path.join(HISTORY_DIR, 'validation.npz')
