import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinatedCam(nn.Module):
    def __init__(self, args, encoder, decoder, src_dense, trg_dense, DEVICE):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = nn.Linear(args.d_model, args.decoder_output_size)

        self.to(DEVICE)

    def forward(self, src, trg):
        '''
        src:  (batch_size, N, T_in, F_in)
        trg: (batch, N, T_out, F_out)
        '''
        encoder_output = self.encode(src)  # (batch_size, N, T_in, F_in)
        return self.decode(trg, encoder_output)  # (batch_size, N, T_out, F_out)

    def encode(self, src):
        '''
        src: (batch_size, N, T_in, F_in)
        '''
        h = self.src_embed(src)   # h: [B, N, T_in, d_model]

        return self.encoder(h)

    def decode(self, trg, encoder_output):
        return self.prediction_generator(self.decoder(self.trg_embed(trg), encoder_output))