import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Transformer(nn.Module):
    def __init__(self, N, d_dec, d_ff, vocab_size, d_enc=512, dropout=0.1):
        super().__init__()
        self.N = N
        self.d_dec = d_dec
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.d_enc = d_enc
        self.dropout = dropout
        self.n_pix = 64
        self.tgt_length = 350
        self.pad_idx = self.vocab_size - 1

        c = copy.deepcopy
        self.attn = MultiHeadedAttention(h=8, d_dec=self.d_dec)
        self.ff = PositionwiseFeedForward(self.d_dec, self.d_ff, self.dropout)
        self.img_enc_to_dec = nn.Linear(self.d_enc, self.d_dec)
        self.img_projection = nn.Linear(self.n_pix, self.tgt_length)
        self.pos_embed = PositionalEncoding(self.d_dec, self.dropout)
        self.voc_embed = Embeddings(self.d_dec, self.vocab_size)
        self.inchi_embed = nn.Sequential(self.voc_embed, c(self.pos_embed))
        self.decoder = Decoder(AttentionLayer(self.d_dec, c(self.attn), c(self.attn),
                               c(self.ff), self.dropout), N)
        self.generator = Generator(self.d_dec, self.vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, imgs, encoded_inchis, inchi_lengths):
        batch_size = imgs.shape[0]

        ### sort input data in decreasing order
        inchi_lengths, sort_ind = inchi_lengths.squeeze(1).sort(dim=0, descending=True)
        imgs = imgs[sort_ind]
        encoded_inchis = encoded_inchis[sort_ind]

        ### transform image for decoder
        imgs = imgs.contiguous().view(batch_size, self.n_pix, -1)
        imgs = F.relu(self.img_enc_to_dec(imgs))
        imgs = imgs.permute(0, 2, 1)
        imgs = F.relu(self.img_projection(imgs))
        imgs = imgs.permute(0, 2, 1)
        imgs = self.pos_embed(imgs)

        ### transform inchi for decoder
        inchis = encoded_inchis[:,:-1]
        inchi_mask = make_std_mask(inchis, self.pad_idx)
        inchis = self.inchi_embed(inchis)
        decode_lengths = (inchi_lengths - 1).tolist()

        ### send through decoder and generate predictions
        x = self.decoder(inchis, imgs, inchi_mask)
        preds = self.generator(x)
        return preds, encoded_inchis, decode_lengths

################### Decoder Layers ######################

class Decoder(nn.Module):
    "Base transformer decoder architecture"
    def __init__(self, decoder_layer, N):
        super().__init__()
        self.layers = clones(decoder_layer, N)
        self.norm = LayerNorm(decoder_layer.d_dec)

    def forward(self, x, mem, tgt_mask):
        for attn_layer in self.layers:
            x = attn_layer(x, mem, tgt_mask)
        return self.norm(x)

class AttentionLayer(nn.Module):
    def __init__(self, d_dec, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.d_dec = d_dec
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.d_dec, dropout), 3)

    def forward(self, x, mem, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, mem, mem))
        return self.sublayer[2](x, self.feed_forward)

class Generator(nn.Module):
    "Generates token predictions after final decoder layer"
    def __init__(self, d_dec, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_dec, vocab_size)

    def forward(self, x):
        return self.proj(x)

############## Attention and FeedForward ################

class MultiHeadedAttention(nn.Module):
    "Multihead attention implementation (based on Vaswani et al.)"
    def __init__(self, h, d_dec, dropout=0.1):
        "Take in model size and number of heads"
        super().__init__()
        assert d_dec % h == 0
        #We assume d_v always equals d_k
        self.d_k = d_dec // h
        self.h = h
        self.linears = clones(nn.Linear(d_dec, d_dec), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Feedforward implementation"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

############## Embedding Layers ###################

class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Static sinusoidal positional encoding layer"
    def __init__(self, d_model, dropout, max_len=350):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += Variable(self.pe[:, :x.size(1)],
                      requires_grad=False)
        return self.dropout(x)

############## Utility Layers ####################

class TorchLayerNorm(nn.Module):
    "Construct a layernorm module (pytorch)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        return self.bn(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (manual)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))

######## MODEL HELPERS ##########

def clones(module, N):
    """Produce N identical layers (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """Mask out subsequent positions (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention' (adapted from Viswani et al.)"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def trans128_4x(vocab_size, N=3):
    return Transformer(N=N, d_dec=128, d_ff=512, vocab_size=vocab_size)

def trans256_4x(vocab_size, N=3):
    return Transformer(N=N, d_dec=256, d_ff=1024, vocab_size=vocab_size)

def trans512_4x(vocab_size, N=3):
    return Transformer(N=N, d_dec=512, d_ff=2048, vocab_size=vocab_size)
