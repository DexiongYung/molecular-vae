import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import device


def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss


class MolecularVAE(nn.Module):
    def __init__(self, vocab: dict, sos_idx: int, pad_idx: int, args):
        super(MolecularVAE, self).__init__()
        self.max_name_len = args.max_len
        self.encoder_mlp_size = args.mlp_encod
        self.latent_size = args.latent
        self.num_layers = args.num_layers
        self.embed_dim = args.word_embed
        self.conv_in_c = args.conv_in_c
        self.conv_out_c = args.conv_out_c
        self.conv_kernals = args.conv_kernals
        self.vocab_size = len(vocab)
        self.eps = args.eps

        self.conv_1 = nn.Conv1d(max_name_len, conv_out_c=[
                                0], kernel_size=conv_kernals[0])
        self.conv_2 = nn.Conv1d(conv_in_c=[0], conv_out_c=[
                                1], kernel_size=conv_kernals[1])
        self.conv_3 = nn.Conv1d(conv_in_c=[0], conv_out_c=[
                                2], kernel_size=conv_kernals[2])

        self.linear_0 = nn.Linear(270, self.encoder_mlp_size)
        self.linear_1 = nn.Linear(self.encoder_mlp_size, self.latent_size)
        self.linear_2 = nn.Linear(self.encoder_mlp_size, self.latent_size)
        self.linear_3 = nn.Linear(self.latent_size, self.latent_size)

        self.gru = nn.GRU(args.rnn_hidd + args.latent,
                          args.rnn_hidd, args.num_layers, batch_first=True)
        self.gru_last = nn.GRU(args.rnn_hidd + self.embed_dim,
                               args.rnn_hidd, 1, batch_first=True)
        self.linear_4 = nn.Linear(args.rnn_hidd, self.vocab_size)

        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=pad_idx
        )

        self.selu = nn.SELU()
        self.softmax = nn.Softmax()

        nn.init.xavier_normal_(self.linear_0.weight)
        nn.init.xavier_normal_(self.linear_1.weight)
        nn.init.xavier_normal_(self.linear_2.weight)
        nn.init.xavier_normal_(self.linear_3.weight)
        nn.init.xavier_normal_(self.linear_4.weight)

    def encode(self, x):
        x = self.selu(self.conv_1(x))
        x = self.selu(self.conv_2(x))
        x = self.selu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = self.eps * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z, x_idx_tensor: torch.Tensor = None):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_len, 1)
        output, hn = self.gru(z)

        if x_idx_tensor is not None:
            x_embed = self.char_embedder(x_idx_tensor)
            tf_input = torch.cat((output, x_embed), dim=2)
            all_outs = self.gru_last(tf_input)
            out_reshape = all_outs.contiguous().view(-1, output.size(-1))
            y0 = F.softmax(self.linear_4(out_reshape), dim=1)
            y = y0.contiguous().view(all_outs.size(0), -1, y0.size(-1))
            return y
        else:
            batch_sz = z.shape[0]
            char_inputs = torch.LongTensor(
                [self.sos_idx] * batch_sz).to(device)
            embed_char = self.char_embedder(char_inputs)
            y = []
            for i in range(self.max_len):
                input = torch.cat((output[:, i, :], embed_char), dim=1)
                if i == 0:
                    out, hn = self.gru_last(input.unsqueeze(1))
                else:
                    out, hn = self.gru_last(tf_input[:, i, :].unsqueeze(1), hn)

                samples = torch.distributions.Categorical(
                    out.squeeze(1)).sample()
                embed_char = self.char_embedder(samples)

                y.append()

            y = torch.Tensor(y)

            return y

    def forward(self, x, x_idx_tensor: torch.Tensor = None):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z, x_idx_tensor), z_mean, z_logvar
