from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import gzip
import pandas
import string
import numpy as np
import argparse
import os
import torch.optim as optim
from model.MolecularVAE_TF import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import path
from utilities import *


def train(epoch):
    model.train()
    train_loss = []
    batch_num = len(train_loader)
    train_iterator = iter(train_loader)
    train_idx_iterator = iter(train_idx_loader)
    for i in range(batch_num):
        data = next(train_iterator)
        idx_data = next(train_idx_iterator)
        data = data[0].to(device)
        idx_data = idx_data[0].to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(data, idx_data)
        loss = vae_loss(output, data, mean, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()

        if i % save_every == 0:
            torch.save(model.state_dict(), save_path)
            plot_losses(train_loss, filename=f'{sess_name}.png')

    torch.save(model.state_dict(), save_path)
    print('train', np.mean(train_loss) / len(train_loader.dataset))
    return np.mean(train_loss) / len(train_loader.dataset)


data_train, c_to_n_vocab, n_to_c_vocab, max_len, pad_idx, idx_tensor = load_dataset(
    'data/first.csv', True)
data_train = torch.utils.data.TensorDataset(data_train)
idx_data_train = torch.utils.data.TensorDataset(idx_tensor)
train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=250, shuffle=False)
train_idx_loader = torch.utils.data.DataLoader(
    idx_data_train, batch_size=250, shuffle=False)

torch.manual_seed(42)

sess_name = 'test'
save_every = 100
epochs = 10000
weights_folder = 'weight'

if not path.exists(weights_folder):
    os.mkdir(weights_folder)

save_path = f'{weights_folder}/{sess_name}.path.tar'

model = MolecularVAE(max_len, c_to_n_vocab, pad_idx).to(device)
optimizer = optim.Adam(model.parameters())


for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
