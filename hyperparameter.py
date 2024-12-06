"""
MAD-UV Challenge 2025 Baseline Code
Website: https://www.mad-uv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

# Choose one from ['eGeMAPS', 'spec_ds', 'spec_averaged']
feature = 'egemaps'

feature_list = {
    'egemaps': {
        'feature_length': 299,
        'input_dim': 88
    },
    'spec_ds': {
        'feature_length': 1501,
        'input_dim': 768
    },
    'spec_averaged': {
        'feature_length': 59,
        'input_dim': 300
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
        'input_dim': feature_list[feature]['input_dim'],
        'channels': [32, 64, 128],
        'kernel_sizes': [(7, 7), (5, 5), (3, 3)],
        'strides': [(3, 3), (2, 2), (2, 2)],
        'dropout': 0.5
    },
    'training': {
        'seed': 619,
        'device': 'cuda:0',
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 24,
        'num_epochs': 500,
        'valid_after_epoch': 2
    }
}
