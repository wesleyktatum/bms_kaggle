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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.transforms import Compose, Normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    if args.checkpoint_fn is not None:
        ckpt, args, start_epoch = load_model_from_ckpt(args.checkpoint_fn)
    else:
        ckpt = None
        start_epoch = 0

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

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.model_name is not None:
        args.log_fn = '{}/log{}.txt'.format(args.log_dir, '_'+args.model_name)
    else:
        args.log_fn = '{}/log.txt'.format(args.log_dir)
    try:
        f = open(args.log_fn, 'r')
        f.close()
        already_wrote = True
    except FileNotFoundError:
        already_wrote = False
    log_file = open(args.log_fn, 'a')
    if not already_wrote:
        log_file.write('epoch,batch_idx,data_type,loss,enc_grad_norm,dec_grad_norm,run_time\n')
    log_file.close()
    if args.make_grad_gif:
        os.makedirs('{}_gif'.format(args.model_name), exist_ok=True)
        args.images = []

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
        decoder = trans128_4x(vocab_size=vocab_size, d_enc=d_enc, N=args.n_decoder_layers)
    elif args.decoder == 'trans256_4x':
        decoder = trans256_4x(vocab_size=vocab_size, d_enc=d_enc, N=args.n_decoder_layers)
    elif args.decoder == 'trans512_4x':
        decoder = trans512_4x(vocab_size=vocab_size, d_enc=d_enc, N=args.n_decoder_layers)
    model = CaptionModel(encoder, decoder)

    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=args.encoder_lr,
                                         weight_decay=1e-6)
    encoder_scheduler = CosineAnnealingLR(encoder_optimizer, T_max=4, eta_min=1e-6,
                                          last_epoch=-1)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=args.decoder_lr,
                                         weight_decay=1e-6)
    decoder_scheduler = CosineAnnealingLR(decoder_optimizer, T_max=4, eta_min=1e-6,
                                          last_epoch=-1)

    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        encoder_optimizer.load_state_dict(ckpt['enc_optimizer_state_dict'])
        encoder_scheduler.load_state_dict(ckpt['enc_scheduler_state_dict'])
        decoder_optimizer.load_state_dict(ckpt['dec_optimizer_state_dict'])
        decoder_scheduler.load_state_dict(ckpt['dec_scheduler_state_dict'])
    optimizers = [encoder_optimizer, decoder_optimizer]
    schedulers = [encoder_scheduler, decoder_scheduler]
    model = model.to(DEVICE)

    n_train_shards = get_n_shards(train_dir)
    n_val_shards = get_n_shards(val_dir)

    for epoch in range(start_epoch, start_epoch+args.n_epochs):
        mode = 'train'
        train_shard_ids = np.random.choice(np.arange(n_train_shards), size=1,
                                           replace=False)
        train_losses = []
        batch_counter = 0
        for shard_id in train_shard_ids:
            mol_train = MoleculeDataset(mode, shard_id, args.imgs_dir, args.img_size,
                                        args.prerotated, args.rotate)
            train_loader = torch.utils.data.DataLoader(mol_train, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=0,
                                                       pin_memory=False, drop_last=True)
            train_loss, batch_counter = train(train_loader, model, optimizers, epoch, args,
                                              batch_counter=batch_counter)
            train_losses.append(train_loss)
            del mol_train, train_loader
        train_loss = np.mean(train_losses)

        # mode = 'val'
        # val_shard_ids = np.random.choice(np.arange(n_val_shards), size=1,
        #                                  replace=False)
        # val_losses = []
        # batch_counter = 0
        # for shard_id in val_shard_ids:
        #     val_train = MoleculeDataset(mode, shard_id, args.imgs_dir, args.img_size,
        #                                 args.prerotated, args.rotate)
        #     val_loader = torch.utils.data.DataLoader(val_train, batch_size=args.batch_size,
        #                                              shuffle=True, num_workers=0,
        #                                              pin_memory=False, drop_last=True)
        #     val_loss, batch_counter = validate(val_loader, model, epoch, args,
        #                                        batch_counter=batch_counter)
        #     val_losses.append(val_loss)
        #     del val_train, val_loader
        # val_loss = np.mean(val_losses)
        # print('Epoch - {} Train - {}, Val - {}'.format(epoch, train_loss, val_loss))

        encoder_scheduler.step()
        decoder_scheduler.step()

        if (epoch+1) % args.save_freq == 0:
            epoch_str = str(epoch+1)
            while len(epoch_str) < 3:
                epoch_str = '0' + epoch_str
            if args.model_name is not None:
                save_fn = os.path.join(args.save_dir, 'model_'+args.model_name+'_'+epoch_str+'.ckpt')
            else:
                save_fn = os.path.join(args.save_dir, 'model_'+epoch_str+'.ckpt')
            save(model, optimizers, schedulers, args, epoch+1, save_fn)

    if args.make_grad_gif:
        imageio.mimsave('{}_grads.gif'.format(args.model_name), args.images)
        shutil.rmtree('{}_gif'.format(args.model_name))


def train(train_loader, model, optimizers, epoch, args, batch_counter=0):

    for optimizer in optimizers:
        optimizer.zero_grad()
    model.train()
    start_time = perf_counter()
    losses = []
    ##############
    # data_load_times = []
    # chunk_times = []
    # to_cuda_times = []
    # model_forward_times = []
    # postprocess_times = []
    # calc_loss_times = []
    # backprop_times = []
    # optimizer_times = []
    # write_log_times = []
    # data_load_start = perf_counter()

    for i, (batch_imgs, batch_encoded_inchis, batch_inchi_lengths) in enumerate(train_loader):
        if i > 1:
            break
        else:
            # to_cuda_start = perf_counter()
            batch_imgs = batch_imgs.to(DEVICE)
            batch_encoded_inchis = batch_encoded_inchis.to(DEVICE)
            batch_inchi_lengths = batch_inchi_lengths.unsqueeze(1).to(DEVICE)
            # to_cuda_end = perf_counter()
            # to_cuda_times.append(to_cuda_end - to_cuda_start)
            # data_load_end = perf_counter()
            # data_load_times.append(data_load_end - data_load_start)
            avg_losses = []
            for j in range(args.batch_chunks):
                # chunk_start = perf_counter()
                imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
                encoded_inchis = batch_encoded_inchis[j*args.chunk_size:(j+1)*args.chunk_size,:]
                inchi_lengths = batch_inchi_lengths[j*args.chunk_size:(j+1)*args.chunk_size,:]
                # chunk_end = perf_counter()
                # chunk_times.append(chunk_end - chunk_start)

                # model_forward_start = perf_counter()
                preds, encoded_inchis, decode_lengths = model(imgs, encoded_inchis, inchi_lengths)
                # model_forward_end = perf_counter()
                # model_forward_times.append(model_forward_end - model_forward_start)

                # postprocess_start = perf_counter()
                targets = encoded_inchis[:,1:]

                preds = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                # postprocess_end = perf_counter()
                # postprocess_times.append(postprocess_end - postprocess_start)

                # calc_loss_start = perf_counter()
                loss = ce_loss(targets, preds, args.char_weights)
                loss /= args.batch_chunks
                # calc_loss_end = perf_counter()
                # calc_loss_times.append(calc_loss_end - calc_loss_start)

                # backprop_start = perf_counter()
                loss.backward()
                # backprop_end = perf_counter()
                # backprop_times.append(backprop_end - backprop_start)

                avg_losses.append(loss.item())

            if args.grad_clip is not None:
                clip_gradient(optimizer, args.grad_clip)

            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), args.grad_clip)
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.grad_clip)

            if args.make_grad_gif:
                grads = plot_grad_flow(model.named_parameters())
                grads.savefig('{}_gif/{}_{}.png'.format(args.model_name, epoch, batch_counter))
                grads.close()
                args.images.append(imageio.imread('{}_gif/{}_{}.png'.format(args.model_name, epoch, batch_counter)))

            # optimizer_start = perf_counter()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            # optimizer_end = perf_counter()
            # optimizer_times.append(optimizer_end - optimizer_start)
            ############################

            stop_time = perf_counter()
            batch_time = round(stop_time - start_time, 5)
            avg_loss = round(np.mean(avg_losses), 5)
            losses.append(avg_loss)
            batch_counter += 1

            # Log
            # write_log_start = perf_counter()
            log_file = open(args.log_fn, 'a')
            log_file.write('{},{},{},{},{},{},{}\n'.format(epoch,
                                                     batch_counter,
                                                     'train',
                                                     avg_loss,
                                                     encoder_grad_norm,
                                                     decoder_grad_norm,
                                                     batch_time))
            log_file.close()
            # write_log_end = perf_counter()
            # write_log_times.append(write_log_end - write_log_start)

            start_time = perf_counter()
            # data_load_start = perf_counter()

    train_loss = np.mean(losses)
    # data_load_time = np.mean(data_load_times)
    # chunk_time = np.mean(chunk_times)
    # to_cuda_time = np.mean(to_cuda_times)
    # model_forward_time = np.mean(model_forward_times)
    # postprocess_time = np.mean(postprocess_times)
    # calc_loss_time = np.mean(calc_loss_times)
    # backprop_time = np.mean(backprop_times)
    # optimizer_time = np.mean(optimizer_times)
    # write_log_time = np.mean(write_log_times)
    # print('Data Loading - {} s'.format(data_load_time))
    # print('Chunking - {} s'.format(chunk_time*args.batch_chunks))
    # print('Sending to CUDA - {} s'.format(to_cuda_time))
    # print('Model Forward - {} s'.format(model_forward_time*args.batch_chunks))
    # print('Postprocessing - {} s'.format(postprocess_time*args.batch_chunks))
    # print('Calculating Loss - {} s'.format(calc_loss_time*args.batch_chunks))
    # print('Backpropagating - {} s'.format(backprop_time*args.batch_chunks))
    # print('Optimizer Gradient - {} s'.format(optimizer_time))
    # print('Writing Log - {} s'.format(write_log_time))
    return train_loss, batch_counter

def validate(val_loader, model, epoch, args, batch_counter=0):

    model.eval()
    start_time = perf_counter()
    losses = []

    with torch.no_grad():
        for i, (batch_imgs, batch_encoded_inchis, batch_inchi_lengths) in enumerate(val_loader):
            batch_imgs = batch_imgs.to(DEVICE)
            batch_encoded_inchis = batch_encoded_inchis.to(DEVICE)
            batch_inchi_lengths = batch_inchi_lengths.unsqueeze(1).to(DEVICE)
            avg_losses = []
            for j in range(args.batch_chunks):
                imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
                encoded_inchis = batch_encoded_inchis[j*args.chunk_size:(j+1)*args.chunk_size,:]
                inchi_lengths = batch_inchi_lengths[j*args.chunk_size:(j+1)*args.chunk_size,:]

                preds, encoded_inchis, decode_lengths = model(imgs, encoded_inchis, inchi_lengths)

                targets = encoded_inchis[:,1:]

                preds = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                loss = ce_loss(targets, preds, args.char_weights)

                avg_losses.append(loss.item())

            stop_time = perf_counter()
            batch_time = round(stop_time - start_time, 5)
            avg_loss = round(np.mean(avg_losses), 5)
            losses.append(avg_loss)
            batch_counter += 1

            # Log
            log_file = open(args.log_fn, 'a')
            log_file.write('{},{},{},{},{},{},{}\n'.format(epoch,
                                                     batch_counter,
                                                     'val',
                                                     avg_loss,
                                                     0,
                                                     0,
                                                     batch_time))
            log_file.close()

            start_time = perf_counter()

    val_loss = np.mean(losses)
    return val_loss, batch_counter

def save(model, optimizers, schedulers, args, epoch, save_fn):
    enc_optimizer, dec_optimizer = optimizers
    enc_scheduler, dec_scheduler = schedulers
    save_state = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'enc_optimizer_state_dict': enc_optimizer.state_dict(),
                  'dec_optimizer_state_dict': dec_optimizer.state_dict(),
                  'enc_scheduler_state_dict': enc_scheduler.state_dict(),
                  'dec_scheduler_state_dict': dec_scheduler.state_dict(),
                  'args': args}
    torch.save(save_state, save_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', type=str, default='/gscratch/pfaendtner/orion/mol_translation/data')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--checkpoint_fn', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--max_inchi_length', type=int, default=350)
    parser.add_argument('--img_size', type=int, choices=[64, 128, 256],
                        default=256)
    parser.add_argument('--rotate', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_chunks', type=int, default=16)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=4e-4)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--prerotated', default=False, action='store_true')
    parser.add_argument('--encoder', choices=['resnet18', 'resnet34', 'resnet50'],
                        default='resnet18')
    parser.add_argument('--decoder', choices=['bilstm', 'trans128_4x', 'trans256_4x', 'trans512_4x'],
                        default='trans128_4x')
    parser.add_argument('--n_decoder_layers', type=int, default=3)
    parser.add_argument('--make_grad_gif', default=False, action='store_true')

    args = parser.parse_args()
    main(args)
