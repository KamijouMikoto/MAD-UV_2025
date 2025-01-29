"""
MADUV Challenge 2025 Baseline Code
Website: https://www.maduv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

# Choose one from ['full', 'ultra', 'audi']
feature = 'full'

feature_list = {
    'full': {
        'feature_length': 59
    },
    'ultra': {
        'feature_length': 59
    },
    'audi': {
        'feature_length': 59
    }
}


hparam = {
    'path': {
        'train_path': f"./USV_Feature_{feature}/train",
        'valid_path': f"./USV_Feature_{feature}/valid",
        'test_path': f"./USV_Feature_{feature}/test",
        'tensorboard_path': f"./tb_{feature}/",
        'log_path': f"./log_{feature}.txt",
        'model_path': f"./model_{feature}/"
    },
    'model': {
        'feature_length': feature_list[feature]['feature_length'],
        'channels': [32, 64],
        'kernel_sizes': [(5, 5), (3, 3)],
        'paddings': [(2, 2), (1, 1)],
        'dropout': 0.5
    },
    'training': {
        'seed': 619,
        'device': 'cuda:0',
        'learning_rate': 2e-5,
        'weight_decay': 1e-5,
        'batch_size': 48,
        'num_epochs': 500,
        'valid_after_epoch': 2
    }
}
