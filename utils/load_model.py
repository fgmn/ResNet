import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
from models.resnet_cbam import resnet34_cbam


def GetResnet34_CBAM(num_classes, device):
    model = resnet34_cbam(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model


def GetInitResnet34(pretrained, weight_path, num_classes, device):
    model = resnet34()
    if pretrained:  # 加载预训练模型
        model.load_state_dict(torch.load(weight_path))
    # 定义全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model


def LoadTrainedWeight(model, path):
    model.load_state_dict(torch.load(path))
    return model
