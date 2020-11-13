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
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from os import path


def plot_losses(losses, folder: str = "plot", filename: str = "checkpoint.png"):
    if not path.exists(folder):
        os.mkdir(folder)

    x = list(range(len(losses)))
    plt.plot(x, losses, 'b--', label="Unsupervised Loss")
    plt.title("Loss Progression")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()


def load_dataset(filename, split=True):
    df = pd.read_csv(filename)
    names = df['name'].tolist()
    PAD = ']'
    max_len = len(max(names, key=len))
    chars = string.ascii_letters + PAD
    c_to_n_vocab = dict(zip(chars, range(len(chars))))
    n_to_c_vocab = dict(zip(range(len(chars)), chars))

    pad_idx = c_to_n_vocab[PAD]

    names_output = [(s).ljust(max_len, PAD) for s in names]
    names_output = [list(map(c_to_n_vocab.get, s))for s in names_output]
    names_output = torch.LongTensor(names_output)
    names_output = torch.nn.functional.one_hot(
        names_output, len(chars)).type(torch.FloatTensor)

    return names_output, c_to_n_vocab, n_to_c_vocab, max_len


class MolecularVAE(nn.Module):
    def __init__(self, max_len: int, vocab: dict):
        super(MolecularVAE, self).__init__()
        self.max_len = max_len
        self.conv_1 = nn.Conv1d(max_len, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.linear_0 = nn.Linear(270, 435)
        self.linear_1 = nn.Linear(435, 292)
        self.linear_2 = nn.Linear(435, 292)

        self.linear_3 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, len(vocab))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_len, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar


def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss


def train(epoch):
    model.train()
    train_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(data)
        loss = vae_loss(output, data, mean, logvar)
        loss.backward()
        train_loss.append(loss)
        optimizer.step()

        if batch_idx % save_every == 0:
            torch.save(model.state_dict(), save_path)
            plot_losses(train_loss, filename=f'{sess_name}.png')

    torch.save(model.state_dict(), save_path)
    print('train', np.mean(train_loss) / len(train_loader.dataset))
    return train_loss / len(train_loader.dataset)


data_train, c_to_n_vocab, n_to_c_vocab, max_len = load_dataset(
    'data/first.csv')
data_train = torch.utils.data.TensorDataset(data_train)
train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=250, shuffle=True)

torch.manual_seed(42)

sess_name = 'test'
save_every = 100
epochs = 30
weights_folder = 'weight'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not path.exists(weights_folder):
    os.mkdir(weights_folder)

save_path = f'{weights_folder}/{sess_name}.path.tar'

model = MolecularVAE(max_len, c_to_n_vocab).to(device)
optimizer = optim.Adam(model.parameters())


for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
