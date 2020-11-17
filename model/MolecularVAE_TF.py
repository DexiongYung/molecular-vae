import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import DEVICE


def vae_loss(x_decoded_mean, x, z_mean, z_sd):
    bce_loss = F.binary_cross_entropy(x_decoded_mean, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_sd - z_mean.pow(2) - z_sd)
    return bce_loss + kl_loss


class MolecularVAE(nn.Module):
    def __init__(self, vocab: dict, sos_idx: int, pad_idx: int, args):
        super(MolecularVAE, self).__init__()
        self.max_name_len = args.max_name_length
        self.encoder_mlp_size = args.mlp_encode
        self.latent_size = args.latent
        self.num_layers = args.num_layers
        self.embed_dim = args.word_embed
        self.conv_in_c = args.conv_in_sz
        self.conv_out_c = args.conv_out_sz
        self.conv_kernals = args.conv_kernals
        self.vocab_size = len(vocab)
        self.eps = args.eps

        self.conv_1 = nn.Conv1d(self.max_name_len, self.conv_out_c[
                                0], kernel_size=self.conv_kernals[0])
        self.conv_2 = nn.Conv1d(self.conv_in_c[0], self.conv_out_c[
                                1], kernel_size=self.conv_kernals[1])
        self.conv_3 = nn.Conv1d(self.conv_in_c[1], self.conv_out_c[
                                2], kernel_size=self.conv_kernals[2])

        c1_out_sz = self.vocab_size-(self.conv_kernals[0]) + 1
        c2_out_sz = c1_out_sz-(self.conv_kernals[1]) + 1
        c3_out_sz = self.conv_out_c[2] * \
            ((c2_out_sz-(self.conv_kernals[2])) + 1)

        self.encoder_layer = nn.Linear(c3_out_sz, self.encoder_mlp_size)
        self.mean_layer = nn.Linear(self.encoder_mlp_size, self.latent_size)
        self.sd_layer = nn.Linear(self.encoder_mlp_size, self.latent_size)
        self.decoder_layer_start = nn.Linear(
            self.latent_size, self.latent_size)

        self.gru = nn.LSTM(args.latent,
                           args.rnn_hidd, args.num_layers, batch_first=True)
        self.gru_last = nn.LSTM(args.rnn_hidd + self.embed_dim,
                                args.rnn_hidd, 1, batch_first=True)
        self.decode_layer_final = nn.Linear(args.rnn_hidd, self.vocab_size)

        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=pad_idx
        )

        self.selu = nn.SELU()
        self.softmax = nn.Softmax()

        nn.init.xavier_normal_(self.encoder_layer.weight)
        nn.init.xavier_normal_(self.mean_layer.weight)
        nn.init.xavier_normal_(self.sd_layer.weight)
        nn.init.xavier_normal_(self.decoder_layer_start.weight)
        nn.init.xavier_normal_(self.decode_layer_final.weight)

    def encode(self, x):
        x0 = self.selu(self.conv_1(x))
        x1 = self.selu(self.conv_2(x0))
        x2 = self.selu(self.conv_3(x1))
        x3 = x2.view(x.size(0), -1)
        x4 = F.selu(self.encoder_layer(x3))
        return self.mean_layer(x4), F.softplus(self.sd_layer(x4))

    def sampling(self, z_mean, z_sd):
        epsilon = self.eps * torch.randn_like(z_sd)
        return z_sd * epsilon + z_mean

    def decode(self, z, idx_tensor: torch.Tensor = None):
        z = F.selu(self.decoder_layer_start(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_name_len, 1)
        output, _ = self.gru(z)

        if idx_tensor is not None:
            x_embed = self.char_embedder(idx_tensor)
            tf_input = torch.cat((output, x_embed), dim=2)
            all_outs, _ = self.gru_last(tf_input)
            out_reshape = all_outs.contiguous().view(-1, output.size(-1))
            y0 = F.softmax(self.decode_layer_final(out_reshape), dim=1)
            y = y0.contiguous().view(all_outs.size(0), -1, y0.size(-1))
        else:
            batch_sz = z.shape[0]
            char_inputs = torch.LongTensor(
                [self.sos_idx] * batch_sz).to(DEVICE)
            embed_char = self.char_embedder(char_inputs)
            y = None
            for i in range(self.max_len):
                input = torch.cat((output[:, i, :], embed_char), dim=1)
                if i == 0:
                    out, hn = self.gru_last(input.unsqueeze(1))
                else:
                    out, hn = self.gru_last(input.unsqueeze(1), hn)

                sm_out = F.softmax(self.decode_layer_final(out), dim=1)
                samples = torch.distributions.Categorical(
                    sm_out).sample()

                if i == 0:
                    y = sm_out
                else:
                    y = torch.cat(y, sm_out, dim=1)

                embed_char = self.char_embedder(samples)

                y.append(out)

            y = torch.Tensor(y)

        return y

    def forward(self, x, idx_tensor: torch.Tensor = None):
        z_mean, z_sd = self.encode(x)
        z = self.sampling(z_mean, z_sd)
        return self.decode(z, idx_tensor), z_mean, z_sd
