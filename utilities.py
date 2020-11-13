import matplotlib.pyplot as plt
import pandas as pd
import string
import torch
import os
from os import path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD = ']'
chars = string.ascii_letters + PAD

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


def load_dataset(filename, ret_idx_tensor: bool = False):
    df = pd.read_csv(filename)
    names = df['name'].tolist()
    max_len = len(max(names, key=len))
    c_to_n_vocab = dict(zip(chars, range(len(chars))))
    n_to_c_vocab = dict(zip(range(len(chars)), chars))

    pad_idx = c_to_n_vocab[PAD]

    names_output = [(s).ljust(max_len, PAD) for s in names]
    names_output = [list(map(c_to_n_vocab.get, s))for s in names_output]
    idx_tensor = torch.LongTensor(names_output)
    names_output = torch.nn.functional.one_hot(
        idx_tensor, len(chars)).type(torch.FloatTensor)
    
    if ret_idx_tensor:
        return names_output, c_to_n_vocab, n_to_c_vocab, max_len, pad_idx, idx_tensor
    else:    
        return names_output, c_to_n_vocab, n_to_c_vocab, max_len, pad_idx

def load_vocab():
    c_to_n_vocab = dict(zip(chars, range(len(chars))))
    n_to_c_vocab = dict(zip(range(len(chars)), chars))
    return c_to_n_vocab, n_to_c_vocab, PAD
