import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import imageio
from time import perf_counter
import argparse
import numpy as np
import pandas as pd

from util import *
from loss import ce_loss
# from dataloader import MoleculeDataset
from models.sasa import ResNet26, ResNet38, ResNet50
from models.axial import axial18s, axial18srpe, axial26s, axial50s, axial50m, axial50l
from models.resnet import resnet18, resnet34, resnet50
from models.bilstm import biLSTM512
from models.transformer import trans128_4x, trans256_4x, trans512_4x
from models.caption import CaptionModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import Levenshtein as lev

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    train_dir = os.path.join(args.imgs_dir, 'train_shards')
    val_dir = os.path.join(args.imgs_dir, 'val_shards')
    with open('{}/char_dict.json'.format(args.data_dir), 'r') as f:
        char_dict = json.load(f)
    with open('{}/ord_dict.json'.format(args.data_dir), 'r') as f:
        ord_dict = json.load(f)
    char_weights = torch.tensor(np.load('{}/char_weights.npy'.format(args.data_dir))).float()
    char_weights = char_weights.to(DEVICE)
    vocab_size = len(char_dict.keys())
    args.char_weights = char_weights
    args.chunk_size = args.batch_size // args.batch_chunks

    os.makedirs(args.range_test_dir, exist_ok=True)
    if args.model_name is not None:
        args.range_fn = '{}/{}_lr_range_test.txt'.format(args.range_test_dir, '_'+args.model_name)
    else:
        args.range_fn = '{}/{}_lr_range_test.txt'.format(args.range_test_dir)
    range_file = open(args.range_fn, 'a')
    range_file.write('lr,lev_dist\n')
    range_file.close()

    if args.encoder == 'resnet18':
        encoder = resnet18(pretrained=False, finetune=True)
        d_enc = 512
    elif args.encoder == 'resnet34':
        encoder = resnet34(pretrained=False, finetune=True)
        d_enc = 512
    elif args.encoder == 'resnet50':
        encoder = resnet50(pretrained=False, finetune=True)
        d_enc = 2048
    if args.decoder == 'bilstm':
        decoder = biLSTM512(vocab_size=vocab_size, device=DEVICE, d_enc=d_enc)
    elif args.decoder == 'trans128_4x':
        decoder = trans128_4x(vocab_size=vocab_size, d_enc=d_enc, device=DEVICE,
                              N=args.n_decoder_layers, teacher_force=True)
    elif args.decoder == 'trans256_4x':
        decoder = trans256_4x(vocab_size=vocab_size, d_enc=d_enc, device=DEVICE,
                              N=args.n_decoder_layers, teacher_force=True)
    elif args.decoder == 'trans512_4x':
        decoder = trans512_4x(vocab_size=vocab_size, d_enc=d_enc, device=DEVICE,
                              N=args.n_decoder_layers, teacher_force=True)
    model = CaptionModel(encoder, decoder)
    model = model.to(DEVICE)
    if args.test_encoder_lr:
        ranged_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=args.base_lr)
        const_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=args.decoder_lr,
                                           weight_decay=1e-6)
    elif args.test_decoder_lr:
        const_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=args.encoder_lr,
                                           weight_decay=1e-6)
        ranged_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=args.base_lr)
    optimizers = [ranged_optimizer, const_optimizer]

    n_train_shards = get_n_shards(train_dir)
    n_val_shards = get_n_shards(val_dir)

    lr_range = np.geomspace(args.base_lr, args.max_lr, args.n_lr)
    for lr in lr_range:
        for p in self.optimizer.param_groups:
            p['lr'] = lr

        mode = 'train'
        train_shard_id = np.random.choice(np.arange(n_train_shards-1), size=1,
                                          replace=False)[0]
        mol_train = MoleculeDataset(mode, shard_id, args.imgs_dir, args.img_size,
                                    args.prerotated, args.rotate)
        train_loader = torch.utils.data.DataLoader(mol_train, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=0,
                                                   pin_memory=False, drop_last=True)
        train(train_loader, model, optimizers, args)
        del mol_train, train_loader

        mode = 'val'
        val_shard_id = 0
        val_train = MoleculeDataset(mode, shard_id, args.imgs_dir, args.img_size,
                                    args.prerotated, args.rotate)
        val_loader = torch.utils.data.DataLoader(val_train, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False, drop_last=True)
        avg_lev_dist = validate(val_loader, model, args)
        del val_train, val_loader

        range_file = open(args.range_fn, 'a')
        range_file.write('{},{}\n'.format(lr, avg_lev_dist))
        range_file.close()


def train(train_loader, model, optimizers, args):

    for optimizer in optimizers:
        optimizer.zero_grad()
    model.train()

    for i, (batch_imgs, batch_encoded_inchis, batch_inchi_lengths) in enumerate(train_loader):
        batch_imgs = batch_imgs.to(DEVICE)
        batch_encoded_inchis = batch_encoded_inchis.to(DEVICE)
        batch_inchi_lengths = batch_inchi_lengths.unsqueeze(1).to(DEVICE)
        for j in range(args.batch_chunks):
            imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
            encoded_inchis = batch_encoded_inchis[j*args.chunk_size:(j+1)*args.chunk_size,:]
            inchi_lengths = batch_inchi_lengths[j*args.chunk_size:(j+1)*args.chunk_size,:]

            preds, encoded_inchis, decode_lengths = model(imgs, encoded_inchis, inchi_lengths, args.mix_scheduler)
            targets = encoded_inchis[:,1:]

            preds = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = ce_loss(targets, preds, args.char_weights)
            loss.backward()

        if args.grad_clip is not None:
            clip_gradient(optimizer, args.grad_clip)

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

def validate(val_loader, model, args):

    model.eval()
    lev_dists = []
    n_samples_evaluated = 0

    with torch.no_grad():
        for i, (batch_imgs, batch_encoded_inchis, batch_inchi_lengths) in enumerate(val_loader):
            batch_imgs = batch_imgs.to(DEVICE)
            batch_encoded_inchis = batch_encoded_inchis.to(DEVICE)
            batch_inchi_lengths = batch_inchi_lengths.unsqueeze(1).to(DEVICE)
            batch_lev_dists = []
            for j in range(args.batch_chunks):
                imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
                encoded_inchis = batch_encoded_inchis[j*args.chunk_size:(j+1)*args.chunk_size,:]
                inchi_lengths = batch_inchi_lengths[j*args.chunk_size:(j+1)*args.chunk_size,:]

                decoded = model.predict(imgs, search_mode='greedy', width=1, device=DEVICE).cpu()
                for k in range(args.chunk_size):
                    pred_inchi = decode_inchi(decoded[k,:], ord_dict)
                    true_inchi = decode_inchi(encoded_inchis[k,1:], ord_dict)
                    lev_dist = lev.distance(pred_inchi, true_inchi)
                    batch_lev_dists.append(lev_dist)
            n_samples_evaluated += args.batch_size
            lev_dists.append(np.mean(batch_lev_dists))
            if n_samples_evaluated > args.n_val_samples:
                break

    avg_lev_dist = np.mean(lev_dists)
    return avg_lev_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', type=str, default='/gscratch/pfaendtner/orion/mol_translation/data')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--range_test_dir', type=str, default='range_tests')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--max_inchi_length', type=int, default=350)
    parser.add_argument('--img_size', type=int, choices=[64, 128, 256],
                        default=256)
    parser.add_argument('--rotate', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_chunks', type=int, default=16)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=4e-4)
    parser.add_argument('--test_encoder_lr', default=False, action='store_true')
    parser.add_argument('--test_decoder_lr', default=False, action='store_true')
    parser.add_argument('--max_lr', type=float, default=1e-1)
    parser.add_argument('--base_lr', type=float, default=5e-6)
    parser.add_argument('--n_lr', type=int, default=20)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--n_val_samples', type=int, default=10000)
    parser.add_argument('--grad_clip', type=int, default=5)
    parser.add_argument('--prerotated', default=False, action='store_true')
    parser.add_argument('--encoder', choices=['resnet18', 'resnet34', 'resnet50'],
                        default='resnet34')
    parser.add_argument('--decoder', choices=['bilstm', 'trans128_4x', 'trans256_4x', 'trans512_4x'],
                        default='trans512_4x')
    parser.add_argument('--n_decoder_layers', type=int, default=3)

    args = parser.parse_args()
    main(args)
