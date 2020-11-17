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
from model.MolecularVAE import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import path
from utilities import *

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Session name', type=str, default='new_eps')
parser.add_argument('--max_name_length',
                    help='Max name generation length', type=int, default=40)
parser.add_argument('--batch_size', help='batch_size', type=int, default=100)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--RNN_hidden_size',
                    help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--word_embed',
                    help='Word embedding size', type=int, default=200)
parser.add_argument(
    '--num_layers', help='number of rnn layer', type=int, default=3)
parser.add_argument('--num_epochs', help='epochs', type=int, default=1000)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument(
    '--percent_train', help='Percent of the data used for training', type=float, default=0.75)
parser.add_argument('--name_file', help='CSVs of names for training and testing',
                    type=str, default='data/first.csv')
parser.add_argument('--weight_dir', help='save dir',
                    type=str, default='weight/')
parser.add_argument('--save_every',
                    help='Number of iterations before saving', type=int, default=200)
parser.add_argument('--continue_train',
                    help='Continue training', type=bool, default=False)
args = parser.parse_args()


def train(epoch):
    model.train()
    train_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(data)
        loss = vae_loss(output, data, mean, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()

        if batch_idx % save_every == 0:
            torch.save(model.state_dict(), save_path)
            plot_losses(train_loss, filename=f'{sess_name}.png')

    torch.save(model.state_dict(), save_path)
    print('train', np.mean(train_loss) / len(train_loader.dataset))
    return np.mean(train_loss) / len(train_loader.dataset)


data_train, c_to_n_vocab, n_to_c_vocab, max_len, pad_idx = load_dataset(
    'data/first.csv')
data_train = torch.utils.data.TensorDataset(data_train)
train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=150, shuffle=True)

torch.manual_seed(42)

sess_name = 'no_tf'
save_every = 100
epochs = 10000
weights_folder = 'weight'

if not path.exists(weights_folder):
    os.mkdir(weights_folder)

save_path = f'{weights_folder}/{sess_name}.path.tar'

model = MolecularVAE(max_len, c_to_n_vocab).to(device)
# model.load_state_dict(torch.load('weight/test.path.tar'))
optimizer = optim.Adam(model.parameters(), lr=1e-20)


for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
