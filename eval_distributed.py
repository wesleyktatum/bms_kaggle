import os
import json
import argparse
import numpy as np
import pandas as pd
from time import perf_counter

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

import torch.multiprocessing as mp
import torch.distributed as dist

import Levenshtein as lev

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(gpu, args, shard_id):
    rank = gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.n_gpus, rank=rank)
    ckpt, ckpt_args, _ = load_model_from_ckpt(args.checkpoint_fn)

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
        decoder = biLSTM512(vocab_size=args.vocab_size, device=DEVICE, d_enc=d_enc)
    elif ckpt_args.decoder == 'trans128_4x':
        decoder = trans128_4x(vocab_size=args.vocab_size, d_enc=d_enc, N=ckpt_args.n_decoder_layers,
                              device=DEVICE, teacher_force=False)
    elif ckpt_args.decoder == 'trans256_4x':
        decoder = trans256_4x(vocab_size=args.vocab_size, d_enc=d_enc, N=ckpt_args.n_decoder_layers,
                              device=DEVICE, teacher_force=False)
    elif ckpt_args.decoder == 'trans512_4x':
        decoder = trans512_4x(vocab_size=args.vocab_size, d_enc=d_enc, N=ckpt_args.n_decoder_layers,
                              device=DEVICE, teacher_force=False)
    model = CaptionModel(encoder, decoder)
    model.load_state_dict(ckpt['model_state_dict'])
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.eval()

    if args.unrotated:
        write_fn = os.path.join(args.write_dir, '{}_{}_{}_unrotated_predictions{}.txt'.format(args.checkpoint_fn.split('/')[-1].split('.')[0], args.mode, args.search_mode, gpu))
    else:
        write_fn = os.path.join(args.write_dir, '{}_{}_{}_rotated_predictions{}.txt'.format(args.checkpoint_fn.split('/')[-1].split('.')[0], args.mode, args.search_mode, gpu))
    try:
        f = open(write_fn, 'r')
        f.close()
        already_wrote = True
    except FileNotFoundError:
        already_wrote = False
    if not already_wrote:
        log_file = open(write_fn, 'a')
        log_file.write('image_id\tInChI\n')
        log_file.close()

    print('loading shard {}...'.format(shard_id))
    mol_data = MoleculeDataset(args.mode, shard_id, args.imgs_dir, ckpt_args.img_size, unrotated=args.unrotated, rotate=False)
    img_ids = pd.read_csv(os.path.join(args.imgs_dir, '{}_shards'.format(args.mode), 'img_id_shard{}.csv'.format(shard_id))).image_id.values

    data_sampler = torch.utils.data.distributed.DistributedSampler(mol_data,
                                                                   num_replicas=args.n_gpus,
                                                                   rank=rank,
                                                                   shuffle=False)

    data_loader = torch.utils.data.DataLoader(mol_data, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0,
                                              pin_memory=False, drop_last=False,
                                              sampler=data_sampler)

    for i, (batch_imgs, batch_img_id_idxs) in enumerate(data_loader):
        batch_imgs = batch_imgs.cuda(non_blocking=True)
        batch_size = batch_imgs.shape[0]
        final_batch = False
        if batch_size != args.batch_size:
            final_batch = True
            final_batch_size = batch_size
            full_chunks = int(final_batch_size / args.chunk_size)
            final_chunk = final_batch_size % args.chunk_size
        if final_batch:
            for j in range(full_chunks+1):
                if j == full_chunks:
                    imgs = batch_imgs[j*args.chunk_size:(j*args.chunk_size)+final_chunk,:,:,:]
                    img_id_idxs = batch_img_id_idxs[j*args.chunk_size:(j*args.chunk_size)+final_chunk]
                    decoded = model.module.predict(imgs, search_mode=args.search_mode, width=args.beam_width,
                                                   device=DEVICE)
                    for k, img_id_idx in enumerate(img_id_idxs):
                        pred_inchi = decode_inchi(decoded[k,:], args.ord_dict)
                        img_id = img_ids[img_id_idx]
                        log_file = open(write_fn, 'a')
                        log_file.write('{}\t{}\n'.format(img_id, pred_inchi))
                        log_file.close()
                else:
                    imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
                    img_id_idxs = batch_img_id_idxs[j*args.chunk_size:(j+1)*args.chunk_size]
                    decoded = model.module.predict(imgs, search_mode=args.search_mode, width=args.beam_width,
                                                   device=DEVICE)
                    for k, img_id_idx in enumerate(img_id_idxs):
                        pred_inchi = decode_inchi(decoded[k,:], args.ord_dict)
                        img_id = img_ids[img_id_idx]
                        log_file = open(write_fn, 'a')
                        log_file.write('{}\t{}\n'.format(img_id, pred_inchi))
                        log_file.close()
        else:
            for j in range(args.batch_chunks):
                imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
                img_id_idxs = batch_img_id_idxs[j*args.chunk_size:(j+1)*args.chunk_size]
                decoded = model.module.predict(imgs, search_mode=args.search_mode, width=args.beam_width,
                                               device=DEVICE)
                for k, img_id_idx in enumerate(img_id_idxs):
                    pred_inchi = decode_inchi(decoded[k,:], args.ord_dict)
                    img_id = img_ids[img_id_idx]
                    log_file = open(write_fn, 'a')
                    log_file.write('{}\t{}\n'.format(img_id, pred_inchi))
                    log_file.close()
    del mol_data, data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', type=str, default='/gscratch/pfaendtner/orion/mol_translation/data')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--eval_dir', type=str, default='eval')
    parser.add_argument('--search_mode', choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam_width', type=int, default=4)
    parser.add_argument('--checkpoint_fn', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_chunks', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--unrotated', default=False, action='store_true')
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    args.mode = 'eval'
    shards_dir = os.path.join(args.imgs_dir, '{}_shards'.format(args.mode))
    with open('{}/char_dict.json'.format(args.data_dir), 'r') as f:
        args.char_dict = json.load(f)
    with open('{}/ord_dict.json'.format(args.data_dir), 'r') as f:
        args.ord_dict = json.load(f)
    args.vocab_size = len(args.char_dict.keys())
    args.chunk_size = args.batch_size // args.batch_chunks

    os.makedirs(args.eval_dir, exist_ok=True)
    args.n_gpus = torch.cuda.device_count()

    if args.unrotated:
        args.write_dir = os.path.join(args.eval_dir, '{}_{}_{}_unrotated_predictions/'.format(args.checkpoint_fn.split('/')[-1].split('.')[0], args.mode, args.search_mode))
    else:
        args.write_dir = os.path.join(args.eval_dir, '{}_{}_{}_rotated_predictions/'.format(args.checkpoint_fn.split('/')[-1].split('.')[0], args.mode, args.search_mode))
    os.makedirs(args.write_dir, exist_ok=True)

    n_shards = get_n_shards(shards_dir)
    for shard_id in range(n_shards):
        print(shard_id)
        print('crafting spawns...')

        mp.spawn(main, nprocs=args.n_gpus, args=(args, shard_id,))
