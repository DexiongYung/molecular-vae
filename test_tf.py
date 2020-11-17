from model.MolecularVAE_TF import MolecularVAE
from utilities import *
import torch
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--session_name',
                    help='Session name', type=str, default='argsparse')
parser.add_argument('--name',
                    help='Person name to test', type=str, default='Michael')
parser.add_argument('--weight_dir', help='save dir',
                    type=str, default='weight/')
parser.add_argument('--eps', help='error from sampling',
                    type=float, default=1e-2)
args = parser.parse_args()

json_file = json.load(open(f'json/{args.session_name}.json', 'r'))
json_files['eps'] = args.eps
t_args = argparse.Namespace()
t_args.__dict__.update(json_file)
args = parser.parse_args(namespace=t_args)

SOS = args.SOS
PAD = args.PAD
c_to_n_vocab = args.c_to_n_vocab
n_to_c_vocab = args.n_to_c_vocab
sos_idx = args.sos_idx
pad_idx = args.pad_idx
max_len = args.max_name_length


def test(test, idx_tensor):
    model.eval()
    output, mean, logvar = model(test, idx_tensor)
    output = torch.argmax(output, dim=2)
    output = output[0, :].tolist()
    output = ''.join(n_to_c_vocab[n] for n in output)
    return output


model = MolecularVAE(c_to_n_vocab, sos_idx, pad_idx, args).to(DEVICE)
model.load(f'{args.weight_dir}/{args.session_name}')

name = (SOS + args.name).ljust(max_len, PAD)
idx_name = [c_to_n_vocab[s] for s in name]
name = (args.name).ljust(max_len, PAD)
name = [c_to_n_vocab[s] for s in name]
idx_tensor = torch.LongTensor(idx_name).unsqueeze(0).to(DEVICE)
names_output = torch.LongTensor(name).unsqueeze(0)
names_output = torch.nn.functional.one_hot(
    names_output, len(c_to_n_vocab)).type(torch.FloatTensor).to(DEVICE)

print(test(names_output, idx_tensor))
