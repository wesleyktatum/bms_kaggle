import os
import json
import argparse
import numpy as np
import pandas as pd

from util import *
from dataloader import MoleculeDataset
from models.sasa import ResNet26, ResNet38, ResNet50
from models.axial import axial18s, axial18srpe, axial26s, axial50s, axial50m, axial50l
from models.resnet import resnet18, resnet34, resnet50
from models.bilstm import biLSTM512
from models.transformer import trans128_4x, trans256_4x, trans512_4x
from models.caption import CaptionModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import Levenshtein as lev

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    shards_dir = os.path.join(args.imgs_dir, '{}_shards'.format(args.mode))
    with open('{}/char_dict.json'.format(args.data_dir), 'r') as f:
        char_dict = json.load(f)
    with open('{}/ord_dict.json'.format(args.data_dir), 'r') as f:
        ord_dict = json.load(f)
    vocab_size = len(char_dict.keys())
    args.chunk_size = args.batch_size // args.batch_chunks

    os.makedirs(args.eval_dir, exist_ok=True)
    ckpt, ckpt_args, n_epochs = load_model_from_ckpt(args.checkpoint_fn)

    if ckpt_args.encoder == 'resnet18':
        encoder = resnet18(pretrained=False, finetune=True)
        d_enc = 512
    elif ckpt_args.encoder == 'resnet34':
        encoder = resnet34(pretrained=False, finetune=True)
        d_enc = 512
    elif ckpt_args.encoder == 'resnet50':
        encoder = resnet50(pretrained=False, finetune=True)
        d_enc = 2048
    if ckpt_args.decoder == 'bilstm':
        decoder = biLSTM512(vocab_size=vocab_size, device=DEVICE, d_enc=d_enc)
    elif ckpt_args.decoder == 'trans128_4x':
        decoder = trans128_4x(vocab_size=vocab_size, d_enc=d_enc, N=ckpt_args.n_decoder_layers)
    elif ckpt_args.decoder == 'trans256_4x':
        decoder = trans256_4x(vocab_size=vocab_size, d_enc=d_enc, N=ckpt_args.n_decoder_layers)
    elif ckpt_args.decoder == 'trans512_4x':
        decoder = trans512_4x(vocab_size=vocab_size, d_enc=d_enc, N=ckpt_args.n_decoder_layers)
    model = CaptionModel(encoder, decoder)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    if args.mode == 'eval':
        img_ids = pd.read_csv(os.path.join(args.imgs_dir, 'sample_submission.csv')).image_id.values

    if args.mode == 'eval' or args.write_predictions:
        write_fn = os.path.join(args.eval_dir, '{}_{}_predictions.txt'.format(ckpt_args.model_name, n_epochs))
        log_file = open(write_fn, 'a')
        log_file.write('image_id,InChI\n')
        log_file.close()

    n_shards = get_n_shards(shards_dir)

    lev_dists = []
    for shard_id in range(n_shards):
        if args.mode == 'eval':
            if shard_id > 0:
                break
            mol_data = MoleculeDataset(args.mode, shard_id, args.imgs_dir, ckpt_args.img_size, rotate=False)
            data_loader = torch.utils.data.DataLoader(mol_data, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=0,
                                                      pin_memory=False, drop_last=False)
            for i, batch_imgs in enumerate(data_loader):
                if i > 9:
                    break
                batch_imgs = batch_imgs.to(DEVICE)
                for j in range(args.batch_chunks):
                    imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
                    img_id_idx = shard_id*mol_data.shard_size+i*args.batch_size+j*args.chunk_size
                    decoded = model.predict(imgs, search_mode=args.search_mode, width=args.beam_width,
                                            device=DEVICE).cpu()
                    for k in range(args.chunk_size):
                        pred_inchi = decode_inchi(decoded[k,:], ord_dict)
                        img_id = img_ids[img_id_idx+k]
                        log_file = open(write_fn, 'a')
                        log_file.write('{},{}\n'.format(img_id, pred_inchi))
                        log_file.close()
        else:
            mol_data = MoleculeDataset(args.mode, shard_id, args.imgs_dir, ckpt_args.img_size,
                                       ckpt_args.prerotated, ckpt_args.rotate)
            data_loader = torch.utils.data.DataLoader(mol_data, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=0,
                                                      pin_memory=False, drop_last=False)
            for i, (batch_imgs, batch_encoded_inchis, _) in enumerate(data_loader):
                batch_imgs = batch_imgs.to(DEVICE)
                batch_encoded_inchis = batch_encoded_inchis
                batch_lev_dists = []
                for j in range(args.batch_chunks):
                    imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
                    encoded_inchis = batch_encoded_inchis[j*args.chunk_size:(j+1)*args.chunk_size,:]

                    decoded = model.predict(imgs, search_mode=args.search_mode, width=args.beam_width,
                                            device=DEVICE).cpu()
                    for k in range(args.chunk_size):
                        pred_inchi = decode_inchi(decoded[k,:], ord_dict)
                        true_inchi = decode_inchi(encoded_inchis[k,1:], ord_dict)
                        lev_dist = lev.distance(pred_inchi, true_inchi)
                        batch_lev_dists.append(lev_dist)
                        if args.write_predictions:
                            log_file = open(write_fn, 'a')
                            log_file.write('{}\n'.format(pred_inchi))
                            log_file.close()
                lev_dists.append(np.mean(batch_lev_dists))


    if args.mode != 'eval':
        avg_lev_dist = np.mean(lev_dists)
        print('Average Levenshtein Distance ({}) - {}'.format(args.mode, avg_lev_dist))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', type=str, default='/gscratch/pfaendtner/orion/mol_translation/data')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--eval_dir', type=str, default='eval')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'eval'], default='eval')
    parser.add_argument('--search_mode', choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--checkpoint_fn', type=str, default=None)
    parser.add_argument('--write_predictions', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_chunks', type=int, default=8)

    args = parser.parse_args()
    main(args)
