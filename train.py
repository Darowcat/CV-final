import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import CrossEntropyLoss
import cv2
from config import cfg
from dataset import get_dataset
from pathlib import Path
from datetime import datetime

class CAFR_ResNet50(torch.nn.Module):
    def __init__(self):
        super(CAFR_ResNet50, self).__init__()

        self.model = models.resnet50(pretrained=False)

        self.model.fc = torch.nn.Linear(2048, 4025)

    def forward(self, x):
        logits = self.model(x)
        return logits

class Trainer:
    def __init__(self, log_dir):
        '''Initialize the varibles for training
        Args:
            log_dir: (pathlib.Path) the direction used for logging
        '''
        self.log_dir = log_dir

        self.train_dataloader, self.val_dataloader = get_dataset()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")
        self.model = CAFR_ResNet50().to(self.device)
        self.criterion = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.max_epoch = 25

    def run(self):
        training_result_dir = self.log_dir / 'training_result'
        training_result_dir.mkdir(parents=True)
        metrics = {'train_loss': [], 'valid_loss': []}

        for self.epoch in range(self.max_epoch):
            train_loss = self.train()
            valid_loss = self.valid() 
            print('lr:',get_lr(self.optimizer))
            print(f'Epoch {self.epoch:03d}:')
            print('train loss:', train_loss)
            print('valid loss:', valid_loss)
            metrics['train_loss'].append(train_loss)
            metrics['valid_loss'].append(valid_loss)
            if torch.tensor(metrics['valid_loss']).argmin() == self.epoch:
                torch.save(self.model.state_dict(), str(training_result_dir / 'model.pth'))

    def train(self):
        self.model.train()
        loss_steps = []

        for _, (sample, label) in enumerate(self.train_dataloader):
            sample = sample.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            pred_label = self.model(sample)
            
            loss = self.criterion(pred_label, label)
            
            loss.backward()
            
            self.optimizer.step()
            loss_steps.append(loss.detach().item())

        avg_loss = sum(loss_steps) / len(loss_steps)
        return avg_loss

    @torch.no_grad()
    def valid(self):
        self.model.eval()
        loss_steps = []

        for _, (sample, label) in enumerate(self.val_dataloader):
            sample = sample.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            pred_label = self.model(sample)
            loss = self.criterion(pred_label, label)
            loss_steps.append(loss.detach().item())

        avg_loss = sum(loss_steps) / len(loss_steps)
        return avg_loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


log_dir = Path('./runs/') / f'{datetime.now():%b%d%H%M%S}'
log_dir.mkdir(parents=True)
Trainer(log_dir).run()