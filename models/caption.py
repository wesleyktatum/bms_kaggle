from time import perf_counter

import torch
import torch.nn as nn

class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, encoded_inchis, inchi_lengths):
        # start = perf_counter()
        x = self.encoder(img)
        # stop = perf_counter()
        # encoder_time = stop - start
        # start = perf_counter()
        preds, encoded_inchis, decode_lengths = self.decoder(x, encoded_inchis, inchi_lengths)
        # stop = perf_counter()
        # decoder_time = stop - start
        # log_file = open('logs/log_captionmodel_time.txt', 'a')
        # log_file.write('{},{}\n'.format(encoder_time, decoder_time))
        # log_file.close()
        return preds, encoded_inchis, decode_lengths

    def predict(self, img, search_mode, width, device):
        x = self.encoder(img)
        preds = self.decoder.predict(x, search_mode, width)
        return preds
