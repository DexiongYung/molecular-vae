import matplotlib.pyplot as plt
import pandas as pd
import string
import torch
import os
from os import path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def load_dataset(filename: str, max_len: int, c_to_n_vocab: dict, SOS: str, PAD: str, return_idx: bool = False):
    df = pd.read_csv(filename).iloc[0:5000]
    df = df[df['name'].str.len() <= max_len]
    names = df['name'].tolist()
    vocab_len = len(c_to_n_vocab)
    max_len = max_len

    names_output = [(s).ljust(max_len, PAD) for s in names]
    names_output = [list(map(c_to_n_vocab.get, s))for s in names_output]
    idx_tensor = torch.LongTensor(names_output)
    names_output = torch.nn.functional.one_hot(
        idx_tensor, vocab_len).type(torch.FloatTensor)

    if return_idx:
        names_idx = []

        for name in names:
            name = SOS + name

            if len(name) > max_len:
                name = name[:-1]
                names_idx.append(list(map(c_to_n_vocab.get, name)))
            else:
                names_idx.append(
                    list(map(c_to_n_vocab.get, (name).ljust(max_len, PAD))))

        idx_tensor = torch.LongTensor(names_idx)
        return names_output, idx_tensor
    else:
        return names_output


def get_data_and_probs(n: str, c_to_n_vocab: dict, n_to_c_vocab: dict, SOS: str, PAD: str):
    df = pd.read_csv(n)
    names = df['name'].tolist()
    name_probs = df['probs'].tolist()
    chars = string.ascii_letters + PAD + SOS
    c_to_n_vocab = dict(zip(chars, range(len(chars))))
    n_to_c_vocab = dict(zip(range(len(chars)), chars))

    sos_idx = c_to_n_vocab[SOS]
    pad_idx = c_to_n_vocab[PAD]

    return names, name_probs, c_to_n_vocab, n_to_c_vocab, sos_idx, pad_idx


def create_batch(all_names: list, probs_list: list, batch_size: int, vocab: dict, SOS: str, PAD: str):
    # Name count part of facebook name data and used to create categorical to sample names from to generate batch
    distribution = torch.distributions.Categorical(
        torch.FloatTensor(probs_list))
    names = [all_names[distribution.sample().item()]
             for i in range(batch_size)]

    seq_length = len(max(all_names, key=len))

    names_input = [(s).ljust(seq_length, PAD) for s in names]
    names_input = [list(map(vocab.get, s)) for s in names_input]
    names_input = torch.LongTensor(names_input)
    one_hot = torch.nn.functional.one_hot(
        names_input, len(vocab)).type(torch.FloatTensor)

    names_idx = []
    for name in names:
        name = SOS + name

        if len(name) > seq_length:
            name = name[:-1]
            names_idx.append(list(map(vocab.get, name)))
        else:
            names_idx.append(
                list(map(vocab.get, (name).ljust(seq_length, PAD))))

    return one_hot, torch.LongTensor(names_idx), seq_length, vocab[PAD]
