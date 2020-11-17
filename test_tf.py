from model.MolecularVAE_TF import MolecularVAE
from utilities import *
import torch


def test(test, idx_tensor):
    model.eval()
    output, mean, logvar = model(test, idx_tensor)
    output = torch.argmax(output, dim=2)
    output = output[0, :].tolist()
    output = ''.join(n_to_c_vocab[n] for n in output)
    return output


c_to_n_vocab, n_to_c_vocab, PAD = load_vocab()
pad_idx = c_to_n_vocab[PAD]
max_len = 30

model = MolecularVAE(max_len, c_to_n_vocab, pad_idx).to(device)
model.load_state_dict(torch.load('weight/tf_sample.path.tar'))

name = ('Michael').ljust(max_len, PAD)
idx_name = [c_to_n_vocab[s] for s in name]
name = [c_to_n_vocab[s] for s in name]
idx_tensor = torch.LongTensor(idx_name).unsqueeze(0).to(device)
names_output = torch.LongTensor(name).unsqueeze(0)
names_output = torch.nn.functional.one_hot(
    names_output, len(c_to_n_vocab)).type(torch.FloatTensor).to(device)

print(test(names_output, idx_tensor))
