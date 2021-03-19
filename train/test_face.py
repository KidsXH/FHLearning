from datasets.nutrition import get_nutrition_data_loaders
import numpy as np

if __name__ == '__main__':
    client_names, train_data_loaders, test_data_loaders = get_nutrition_data_loaders(batch_size=100)
    data_loader = train_data_loaders[0]

