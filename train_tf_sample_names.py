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
from model.SampleVAE import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import path
from utilities import *


def train(one_hot, idx_tensor, epoch, batch_num: int = 10000):
    model.train()
    train_loss = []
    for i in range(batch_num):
        optimizer.zero_grad()
        output, mean, logvar = model(one_hot.to(device), idx_tensor.to(device))
        loss = vae_loss(output, one_hot.to(device), mean, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        one_hot, idx_tensor, max_len, pad_idx = create_batch(
            names, name_probs, 300, c_to_n_vocab)

        if i % save_every == 0:
            torch.save(model.state_dict(), save_path)
            plot_losses(train_loss, filename=f'{sess_name}.png')

    torch.save(model.state_dict(), save_path)
    return np.mean(train_loss)


names, name_probs, c_to_n_vocab, n_to_c_vocab, _, _ = get_data_and_probs(
    'data/first.csv')
one_hot, idx_tensor, max_len, pad_idx = create_batch(
    names, name_probs, 400, c_to_n_vocab)

torch.manual_seed(42)

sess_name = 'tf_sample'
save_every = 100
epochs = 10000
weights_folder = 'weight'

if not path.exists(weights_folder):
    os.mkdir(weights_folder)

save_path = f'{weights_folder}/{sess_name}.path.tar'

model = MolecularVAE(max_len, c_to_n_vocab, pad_idx).to(device)
optimizer = optim.Adam(model.parameters(), 1e-30)


for epoch in range(1, epochs + 1):
    train_loss = train(one_hot, idx_tensor, epoch)
