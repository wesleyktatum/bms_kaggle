from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, encoded_inchis, inchi_lengths, mix_scheduler=None):
        x = self.encoder(img)
        preds, encoded_inchis, decode_lengths = self.decoder(x, encoded_inchis, inchi_lengths, mix_scheduler)
        return preds, encoded_inchis, decode_lengths

    def predict(self, img, search_mode, width, device):
        x = self.encoder(img)
        preds = self.decoder.predict(x, search_mode, width, device)
        return preds
