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
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs, optimzer, decay_ratio, current_lr,
                 save_path='./Result/best_resnet34.pth', plot_loss=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = optimzer
        self.decay_sch = [50]
        self.decay_ratio = decay_ratio
        self.current_lr = current_lr
        self.best_acc = 0
        self.log_train_loss = []
        self.log_val_loss = []
        self.save_path = save_path
        self.loss = nn.CrossEntropyLoss()
        self.plot_loss = plot_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for i, (img, label) in enumerate(self.train_loader):
            img = img.cuda()
            label = label.cuda()

            output = self.model(img)

            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            if i % 40 == 0:
                print('Epoch:[', epoch + 1, '/', self.epochs, '][', i + 1, '/', len(self.train_loader), ']',
                      loss.item())

        self.log_train_loss.append(train_loss)

    def val_one_epoch(self, epoch):
        val_loss = 0
        self.model.eval()
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        with torch.no_grad():
            for i, (img, label) in enumerate(self.val_loader):
                img = img.cuda()
                label = label.cuda()

                output = self.model(img)

                loss = self.loss(output, label)
                val_loss += loss.item()

                prediction = torch.argmax(output, 1)
                correct += (prediction == label).sum().float()
                total += len(label)

        acc = (correct / total).cpu().detach().data.numpy()
        print('Epoch: ', epoch + 1, ' test accuracy: ', acc)
        if acc > self.best_acc:
            print('best accuracy: ', acc)
            self.best_acc = acc
            torch.save(self.model.state_dict(), self.save_path)
        self.log_val_loss.append(val_loss)

    def decay_lr(self, epoch):
        if epoch in self.decay_sch:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_lr*self.decay_ratio, betas=(0.5, 0.99))
            self.current_lr *= self.decay_ratio

    def train(self):
        for epoch in range(self.epochs):
            self.decay_lr(epoch)
            self.train_one_epoch(epoch)
            self.val_one_epoch(epoch)

        if self.plot_loss:
            plotLoss(self.log_train_loss, 'train_loss')
            plotLoss(self.log_val_loss, 'val_loss')


def plotLoss(loss_list, label_name):
    plt.plot(loss_list, label=label_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./Result/' + label_name + '.png')
    plt.show()
