# -*- coding: utf-8 -*-
# Author: Molia Chen

import os
import argparse
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import load_network

parser = argparse.ArgumentParser()
parser.add_argument('--cr', type=int, default=8)
parser.add_argument('--scenario', type=str, default='indoor')
parser.add_argument('--data-root', type=str, default='../data')
args = parser.parse_args()

img_channels = 2
img_height = 32
img_width = 32
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


class NMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x):
        x = x.permute(0, 2, 3, 1)
        x_hat = x_hat.permute(0, 2, 3, 1)
        x_real = torch.reshape(x[:, :, :, 0], (x.shape[0], -1))
        x_imag = torch.reshape(x[:, :, :, 1], (x.shape[0], -1))
        x_hat_real = torch.reshape(x_hat[:, :, :, 0], (x.shape[0], -1))
        x_hat_imag = torch.reshape(x_hat[:, :, :, 1], (x.shape[0], -1))

        power = torch.sum((x_real - 0.5) ** 2 + (x_imag - 0.5) ** 2, dim=1)
        mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, dim=1)
        nmse = torch.mean(mse / power)
        return nmse

class MyDataset(Dataset):
    def __init__(self, train_data):
        self.data = train_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]

if __name__ == '__main__':
    if args.scenario == 'indoor':
        mat = sio.loadmat(os.path.join(args.data_root, 'paper_indoor_H_14_train.mat'))
        x_train = mat['H']
        mat = sio.loadmat(os.path.join(args.data_root, 'paper_indoor_H_14_val.mat'))
        x_val = mat['H']
    else:
        mat = sio.loadmat(os.path.join(args.data_root, 'paper_outdoor_H_14_train.mat'))
        x_train = mat['H']
        mat = sio.loadmat(os.path.join(args.data_root, 'paper_outdoor_H_14_val.mat'))
        x_val = mat['H']
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
    x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))
    eval_set = MyDataset(x_val)
    eval_loader = DataLoader(eval_set, batch_size=128, num_workers=4, shuffle=False, pin_memory=True)

    model = load_network(args.scenario, args.cr).to(device)
    state_dict = torch.load(f'checkpoints/{args.scenario}_{args.cr}.pth', map_location=device)
    model.encoder.load_state_dict(state_dict['encoder'])
    model.decoder.load_state_dict(state_dict['decoder'])

    criterion = NMSE()

    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(eval_loader):
            shape = data.shape
            data = data.to(device)
            out = model(data[:, :, :, :14])
            padding = torch.zeros([*shape[:-1], 32 - 14]) + 0.5
            out = torch.cat([out, padding.cuda()], dim=-1)
            loss = criterion(out, data)
            eval_loss += loss.item()*shape[0]
        loss = eval_loss / len(eval_set)
        print(f'Model name: {args.scenario}_{args.cr}')
        print("NMSE: %.6f, %.2f dB" %(loss, 10*np.log10(loss)))
