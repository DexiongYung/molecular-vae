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
import json
import os
import torch.optim as optim
from model.MolecularVAE_TF import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import path
from utilities import *

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Session name', type=str, default='tf')
parser.add_argument('--max_name_length',
                    help='Max name generation length', type=int, default=30)
parser.add_argument('--batch_size', help='batch_size', type=int, default=100)
parser.add_argument('--latent', help='latent_size', type=int, default=200)
parser.add_argument(
    '--RNN_hidd', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--mlp_encod', help='MLP encoder size',
                    type=int, default=512)
parser.add_argument(
    '--word_embed', help='Word embedding size', type=int, default=200)
parser.add_argument(
    '--num_layers', help='number of rnn layer', type=int, default=3)
parser.add_argument('--num_epochs', help='epochs', type=int, default=1000)
parser.add_argument('--conv_kernals', nargs='+', default=[9, 9, 11])
parser.add_argument('--conv_in_sz', nargs='+', default=[9, 9])
parser.add_argument('--conv_out_sz', nargs='+', default=[9, 9, 10])
parser.add_argument('--eps', help='error from sampling',
                    type=float, default=1e-2)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-8)
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

        if i % args.save_every == 0:
            torch.save(model.state_dict(), save_path)
            plot_losses(train_loss, filename=f'{args.name}.png')

    torch.save(model.state_dict(), save_path)
    print('train', np.mean(train_loss) / len(train_loader.dataset))
    return np.mean(train_loss) / len(train_loader.dataset)


PAD = ']'
SOS = '['
VOCAB = string.ascii_letters + SOS + PAD
c_to_n_vocab = dict(zip(VOCAB, range(len(VOCAB))))
n_to_c_vocab = dict(zip(range(len(VOCAB)), VOCAB))
sos_idx = c_to_n_vocab[SOS]
pad_idx = c_to_n_vocab[PAD]

name_in_out, idx_tensor = load_dataset(
    args.name_file, args.max_name_length, c_to_n_vocab, SOS, PAD, True)
data_train = torch.utils.data.TensorDataset(name_in_out)
idx_data_train = torch.utils.data.TensorDataset(idx_tensor)
train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=args.batch_size, shuffle=False)
train_idx_loader = torch.utils.data.DataLoader(
    idx_data_train, batch_size=args.batch_size, shuffle=False)

torch.manual_seed(42)

if args.continue_train:
    json_file = json.load(open(f'json/{args.name}.json', 'r'))
    t_args = argparse.Namespace()
    t_args.__dict__.update(json_file)
    args = parser.parse_args(namespace=t_args)
    model.load(f'{args.weight_dir}/{args.name}')

    SOS = args.SOS
    PAD = args.PAD
    c_to_n_vocab = args.c_to_n_vocab
    n_to_c_vocab = args.n_to_c_vocab
    sos_idx = args.sos_idx
    pad_idx = args.pad_idx
else:
    args.c_to_n_vocab = c_to_n_vocab
    args.n_to_c_vocab = n_to_c_vocab
    args.sos_idx = sos_idx
    args.pad_idx = pad_idx
    args.PAD = PAD
    args.SOS = SOS

    if not path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)

    if not path.exists('json'):
        os.mkdir('json')

    with open(f'json/{args.name}.json', 'w') as f:
        json.dump(vars(args), f)

if not path.exists(args.weights_dir):
    os.mkdir(args.weights_dir)

save_path = f'{args.weights_dirs}/{args.name}.path.tar'

model = MolecularVAE(c_to_n_vocab, sos_idx, pad_idx).to(device)
optimizer = optim.Adam(model.parameters(), args.lr)

if path.exists(save_path):
    model.load_state_dict(torch.load(save_path))

for epoch in range(1, args.num_epochs + 1):
    train_loss = train(epoch)
