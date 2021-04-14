import os
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from sparse import sparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from util import *

class MoleculeDataset(Dataset):
    """
    PyTorch Dataset class to load molecular images and InChIs
    """
    def __init__(self, mode, shard_id, source_dir, img_size, prerotated=False,
                 rotate=True, p=0.5):
        self.mode = mode
        self.shard_id = shard_id
        if self.mode == 'train' or self.mode == 'eval':
            self.shard_size = 200000
        elif self.mode == 'val' or self.mode == 'test':
            self.shard_size = 25000
        self.img_size = img_size
        self.prerotated = prerotated
        if self.prerotated:
            self.rotate = False
        else:
            self.rotate = rotate
        self.p = p
        if self.prerotated:
            self.sparse_path = os.path.join(source_dir, '{}_shards/prerotated'.format(self.mode), 'shard{}.npz'.format(shard_id))
        else:
            self.sparse_path = os.path.join(source_dir, '{}_shards'.format(self.mode), 'shard{}.npz'.format(shard_id))
        self.sparse_imgs = sparse.load_npz(self.sparse_path)

        if mode != 'eval':
            self.inchi_path = os.path.join(source_dir, '{}_shards'.format(self.mode), 'encoded_inchis.npy')
            self.encoded_inchis = np.load(self.inchi_path)
        else:
            self.img_id_path = os.path.join(source_dir, '{}_shards'.format(self.mode), 'img_id_shard{}.csv'.format(shard_id))
            self.img_ids = pd.read_csv(self.img_id_path)

    def __getitem__(self, i):
        ### grab image
        # start = perf_counter()
        sparse_img = self.sparse_imgs[i,:,:,:]
        # stop = perf_counter()
        # grab_sparse_img = stop - start
        # start = perf_counter()
        img = sparse_img.todense().astype(np.float32)
        # stop = perf_counter()
        # cast_to_dense = stop - start
        img = torch.tensor(img)
        if self.img_size != 256:
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(self.img_size, self.img_size))
            img = img.squeeze(0)
        # start = perf_counter()
        if self.rotate:
            angles = [0, 90, 180, 270]
            angle = np.random.choice(angles, size=1, p=[1 - self.p, self.p / 3, self.p / 3, self.p / 3])
            if angle == 0:
                pass
            elif angle == 90:
                img = torch.rot90(img, 1, [1,2])
            elif angle == 180:
                img = torch.rot90(img, 1, [1,2])
                img = torch.rot90(img, 1, [1,2])
            elif angle == 270:
                img = torch.rot90(img, -1, [1,2])
        # stop = perf_counter()
        # rotate_img = stop - start

        ### grab inchi
        if self.mode != 'eval':
            # start = perf_counter()
            inchi_idx = i + (self.shard_size*self.shard_id)
            inchi_data = torch.tensor(self.encoded_inchis[inchi_idx]).long()
            encoded_inchi = inchi_data[:-1]
            inchi_length = inchi_data[-1]
            # stop = perf_counter()
            # grab_inchi = stop - start
            # log_file = open('logs/log_dataloader_times.txt', 'a')
            # log_file.write('{},{},{},{}\n'.format(grab_sparse_img, cast_to_dense, rotate_img, grab_inchi))
            # log_file.close()
            return img, encoded_inchi, inchi_length
        else:
            img_idx = i
            return img, torch.tensor(img_idx).long()

    def __len__(self):
        return self.sparse_imgs.shape[0]
