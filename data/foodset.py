import torch
import torch.nn as nn
import os

import argparse
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torchvision
from torchvision import datasets
import numpy as np
import random


def GetDataLoader(root, ratio, batch_size, train_transform=None, test_transform=None):
    # 定义数据transform
    if train_transform is None:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])

    if test_transform is None:
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])

    food_set = datasets.ImageFolder(root=root, transform=train_transform)
    train_size = int(len(food_set) * ratio)
    test_size = len(food_set) - train_size

    # 将数据集分隔为训练集以及验证集
    train_set, val_set = torch.utils.data.random_split(food_set, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1)
    return train_loader, val_loader
