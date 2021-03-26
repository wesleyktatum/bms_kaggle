import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from time import perf_counter
import argparse

from util import *
from loss import ce_loss
from dataloader import MoleculeDataset
from models.sasa import ResNet26, ResNet38, ResNet50
from models.axial import axial26s, axial50s, axial50m, axial50l
from models.bilstm import biLSTM512
from models.caption import CaptionModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    train_dir = os.path.join(args.imgs_dir, args.train_dir)
    test_dir = os.path.join(args.imgs_dir, args.test_dir)
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
        log_file.write('epoch,batch_idx,data_type,loss,acc,run_time\n')
    log_file.close()

    if args.checkpoint_fn is not None:
        pass
    else:
        encoder = axial26s()
        decoder = biLSTM512(vocab_size=vocab_size, device=DEVICE)
        model = CaptionModel(encoder, decoder)
        start_epoch = 0

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    train_labels = pd.read_csv('{}/train.csv'.format(args.data_dir))
    val_labels = pd.read_csv('{}/val.csv'.format(args.data_dir))
    n_train_shards = len(os.listdir(os.path.join(args.imgs_dir, 'train_shards')))
    n_val_shards = len(os.listdir(os.path.join(args.imgs_dir, 'val_shards')))

    for epoch in range(start_epoch, start_epoch+args.n_epochs):
        mode = 'train'
        train_shard_ids = np.random.choice(np.arange(n_train_shards), size=n_train_shards,
                                           replace=False)
        train_losses = []
        batch_counter = 0
        for shard_id in train_shard_ids:
            mol_train = MoleculeDataset(train_labels, mode, shard_id, args.imgs_dir,
                                        char_dict, args.max_inchi_length)
            train_loader = torch.utils.data.DataLoader(mol_train, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=0,
                                                       pin_memory=False, drop_last=True)
            train_loss, batch_counter = train(train_loader, model, optimizer, epoch, args,
                                              batch_counter=batch_counter)
            train_losses.append(train_loss)
            del mol_train, train_loader
        train_loss = np.mean(train_losses)

        mode = 'val'
        val_shard_ids = np.random.choice(np.arange(n_val_shards), size=n_val_shards,
                                         replace=False)
        val_losses = []
        batch_counter = 0
        for shard_id in val_shard_ids:
            val_train = MoleculeDataset(val_labels, mode, shard_id, args.imgs_dir,
                                        char_dict, args.max_inchi_length)
            val_loader = torch.utils.data.DataLoader(val_train, batch_size=args.batch_size,
                                                     shuffle=True, num_workers=0,
                                                     pin_memory=False, drop_last=True)
            val_loss, batch_counter = validate(val_loader, model, epoch, args,
                                               batch_counter=batch_counter)
            val_losses.append(val_loss)
            del val_train, val_loader
        val_loss = np.mean(val_losses)
        print('Epoch - {} Train - {}, Val - {}'.format(epoch, train_loss, val_loss))
        if (epoch+1) % args.save_freq == 0:
            epoch_str = str(epoch+1)
            while len(epoch_str) < 3:
                epoch_str = '0' + epoch_str
            if args.model_name is not None:
                save_fn = os.path.join(args.save_dir, 'model_'+args.model_name+epoch_str+'.ckpt')
            else:
                save_fn = os.path.join(args.save_dir, 'model_'+epoch_str+'.ckpt')
            save(model, optimizer, args, epoch+1, save_fn)


def train(train_loader, model, optimizer, epoch, args, batch_counter=0):

    model.train()
    start_time = perf_counter()
    losses = []
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
        # data_load_end = perf_counter()
        # data_load_times.append(data_load_end - data_load_start)
        avg_losses = []
        avg_accs = []
        for j in range(args.batch_chunks):
            # chunk_start = perf_counter()
            imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
            encoded_inchis = batch_encoded_inchis[j*args.chunk_size:(j+1)*args.chunk_size,:]
            inchi_lengths = batch_inchi_lengths[j*args.chunk_size:(j+1)*args.chunk_size]
            # chunk_end = perf_counter()
            # chunk_times.append(chunk_end - chunk_start)
            # to_cuda_start = perf_counter()
            imgs = imgs.to(DEVICE)
            encoded_inchis = encoded_inchis.to(DEVICE)
            inchi_lengths = inchi_lengths.unsqueeze(1).to(DEVICE)
            # to_cuda_end = perf_counter()
            # to_cuda_times.append(to_cuda_end - to_cuda_start)

            # model_forward_start = perf_counter()
            preds, encoded_inchis, decode_lengths, alphas, sort_ind = model(imgs, encoded_inchis, inchi_lengths)
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
            loss += args.alpha_c * ((1. - alphas.sum(dim=1))**2).mean()
            # calc_loss_end = perf_counter()
            # calc_loss_times.append(calc_loss_end - calc_loss_start)

            # backprop_start = perf_counter()
            loss.backward()
            acc = accuracy(preds, targets, 1)
            # backprop_end = perf_counter()
            # backprop_times.append(backprop_end - backprop_start)

            avg_losses.append(loss.item())
            avg_accs.append(acc)

        if args.grad_clip is not None:
            clip_gradient(optimizer, args.grad_clip)

        # optimizer_start = perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        # optimizer_end = perf_counter()
        # optimizer_times.append(optimizer_end - optimizer_start)
        stop_time = perf_counter()
        batch_time = round(stop_time - start_time, 5)
        avg_loss = round(np.mean(avg_losses), 5)
        avg_acc = round(np.mean(avg_accs), 2)
        losses.append(avg_loss)
        batch_counter += 1

        # Log
        # write_log_start = perf_counter()
        log_file = open(args.log_fn, 'a')
        log_file.write('{},{},{},{},{},{}\n'.format(epoch,
                                                    batch_counter,
                                                    'train',
                                                    avg_loss,
                                                    avg_acc,
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
    # print('Chunking - {} s'.format(chunk_time))
    # print('Sending to CUDA - {} s'.format(to_cuda_time))
    # print('Model Forward - {} s'.format(model_forward_time))
    # print('Postprocessing - {} s'.format(postprocess_time))
    # print('Calculating Loss - {} s'.format(calc_loss_time))
    # print('Backpropagating - {} s'.format(backprop_time))
    # print('Optimizer Gradient - {} s'.format(optimizer_time))
    # print('Writing Log - {} s'.format(write_log_time))
    return train_loss, batch_counter

def validate(val_loader, model, epoch, args, batch_counter=0):

    model.eval()
    start_time = perf_counter()
    losses = []

    with torch.no_grad():
        for i, (batch_imgs, batch_encoded_inchis, batch_inchi_lengths) in enumerate(val_loader):
            avg_losses = []
            avg_accs = []
            for j in range(args.batch_chunks):
                imgs = batch_imgs[j*args.chunk_size:(j+1)*args.chunk_size,:,:,:]
                encoded_inchis = batch_encoded_inchis[j*args.chunk_size:(j+1)*args.chunk_size,:]
                inchi_lengths = batch_inchi_lengths[j*args.chunk_size:(j+1)*args.chunk_size]
                imgs = imgs.to(DEVICE)
                encoded_inchis = encoded_inchis.to(DEVICE)
                inchi_lengths = inchi_lengths.unsqueeze(1).to(DEVICE)

                preds, encoded_inchis, decode_lengths, alphas, sort_ind = model(imgs, encoded_inchis, inchi_lengths)

                targets = encoded_inchis[:,1:]

                preds = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                loss = ce_loss(targets, preds, args.char_weights)
                loss += args.alpha_c * ((1. - alphas.sum(dim=1))**2).mean()

                acc = accuracy(preds, targets, 1)
                avg_losses.append(loss.item())
                avg_accs.append(acc)

            stop_time = perf_counter()
            batch_time = round(stop_time - start_time, 5)
            avg_loss = round(np.mean(avg_losses), 5)
            avg_acc = round(np.mean(avg_accs), 2)
            losses.append(avg_loss)
            batch_counter += 1

            # Log
            log_file = open(args.log_fn, 'a')
            log_file.write('{},{},{},{},{},{}\n'.format(epoch,
                                                        i, 'val',
                                                        avg_loss,
                                                        avg_acc,
                                                        batch_time))
            log_file.close()

            start_time = perf_counter()

    val_loss = np.mean(losses)
    return val_loss, batch_counter

def save(model, optimizer, args, epoch, save_fn):
    save_state = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'args': {}}
    for arg in vars(args):
        save_state[arg] = getattr(args, arg)
    torch.save(save_state, save_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', type=str, default='/gscratch/pfaendtner/orion/mol_translation/data')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--train_dir', type=str, default='train_resize')
    parser.add_argument('--test_dir', type=str, default='test_resize')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--checkpoint_fn', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--max_inchi_length', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_chunks', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--alpha_c', type=float, default=1.)

    args = parser.parse_args()
    main(args)
