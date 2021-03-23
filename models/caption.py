import torch
import torch.nn as nn

class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, encoded_inchis, inchi_lengths):
        x = self.encoder(x)
        preds, encoded_inchis, decode_lengths, alphas, sort_ind = self.decoder(x, encoded_inchis, inchi_lengths)
        return preds, encoded_inchis, decode_lengths, alphas, sort_ind
