import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss


class MolecularVAE(nn.Module):
    def __init__(self, max_len: int, vocab: dict, pad_idx: int):
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
        embed_dim = 200
        self.gru_last = nn.GRU(501 + 200, 501, 1, batch_first=True)
        self.linear_4 = nn.Linear(501, len(vocab))
        self.vocab_size = len(vocab)

        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.selu = nn.SELU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        x = self.selu(self.conv_1(x))
        x = self.selu(self.conv_2(x))
        x = self.selu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z, x_idx_tensor):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_len, 1)

        x_embed = self.char_embedder(x_idx_tensor)
        output, hn = self.gru(z)

        tf_input = torch.cat((output, x_embed), dim=2)
        all_outs = None
        for i in range(self.max_len):
            if i == 0:
                out, hn = self.gru_last(tf_input[:,i,:].unsqueeze(1))
                all_outs = out
            else:
                out, hn = self.gru_last(tf_input[:,i,:].unsqueeze(1), hn)
                all_outs = torch.cat((all_outs, out), dim=1)
            
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x, x_idx_tensor):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z, x_idx_tensor), z_mean, z_logvar
