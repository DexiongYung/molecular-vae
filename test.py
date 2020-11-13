from model.MolecularVAE import MolecularVAE
from model.MolecularVAE import *
from utilities import *
import torch


def test(test):
    model.eval()
    test = torch.transpose(test, 0, 1).type(torch.FloatTensor).to(device)
    output, mean, logvar = model(test)
    output = torch.argmax(output, dim=2)
    output = output[0,:].tolist()
    output = ''.join(n_to_c_vocab[n] for n in output)
    return output


c_to_n_vocab, n_to_c_vocab, PAD = load_vocab()
max_len = 30

model = MolecularVAE(max_len, c_to_n_vocab).to(device)
model.load_state_dict(torch.load('weight/test.path.tar'))

name = ('Dylan').ljust(max_len, PAD)
name = [list(map(c_to_n_vocab.get, s))for s in name]
names_output = torch.LongTensor(name)
names_output = torch.nn.functional.one_hot(
    names_output, len(c_to_n_vocab)).type(torch.FloatTensor)

print(test(names_output))
