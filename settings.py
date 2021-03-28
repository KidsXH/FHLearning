import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

DATA_HOME = {
    'mnist': os.path.join(BASE_DIR, 'data', 'mnist'),
    'face': os.path.join(BASE_DIR, 'data', 'face'),
    'cifar10': os.path.join(BASE_DIR, 'data', 'cifar10'),
}

