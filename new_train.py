import torch
import torch.nn as nn
import os

import argparse
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torchvision

import numpy as np
import random
from model import resnet34
from data.foodset import GetDataLoader
from utils.load_model import GetInitResnet34, LoadTrainedWeight, GetResnet34_CBAM
from utils.train_model import Trainer

if __name__ == '__main__':

    # 参数解析模块
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data_set/food_data/food_photos")
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--num_workers", default=4)
    parser.add_argument("--train_val_ratio", default=0.9)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--class_num", default=10)
    parser.add_argument("--decay_ratio", default=0.1)
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--baseline_pretrained_path", default='./resnet34-pre.pth')
    parser.add_argument("--save_path", default='./Result/best_resnet34_cbam.pth')
    parser.add_argument("--plot_loss", default=True)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--use_cuda", default=True)
    parser.add_argument("--model", default="cbam")  # ['baseline', 'cbam']

    args = parser.parse_args()

    if args.use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 数据加载
    train_loader, val_loader = GetDataLoader(args.data_root, args.train_val_ratio, args.batch_size)

    # 模型加载
    if args.model == 'baseline':
        model = GetInitResnet34(args.pretrained, args.baseline_pretrained_path, args.class_num, device)
    elif args.model == 'cbam':
        model = GetResnet34_CBAM(args.class_num, device)
    else:
        raise ValueError("Model selection error.")

    # 优化器选择，训练
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.99))
    trainer = Trainer(model, train_loader, val_loader, args.epochs, optimizer, args.decay_ratio, args.lr,
                      args.save_path, args.plot_loss)

    trainer.train()

