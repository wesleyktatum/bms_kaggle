import os
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from sparse import sparse

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from util import *

class MoleculeDataset(Dataset):
    """
    PyTorch Dataset class to load molecular images and InChIs
    """
    def __init__(self, mode, shard_id, source_dir, rotate=True, p=0.5):
        self.mode = mode
        self.shard_id = shard_id
        self.sparse_path = os.path.join(source_dir, '{}_shards'.format(self.mode), 'shard{}.npz'.format(shard_id))
        self.sparse_imgs = sparse.load_npz(self.sparse_path)
        self.inchi_path = os.path.join(source_dir, '{}_shards'.format(self.mode), 'encoded_inchis.npy')
        self.encoded_inchis = np.load(self.inchi_path)
        self.rotate = rotate
        self.p = p

    def __getitem__(self, i):
        ### grab image
        start = perf_counter()
        sparse_img = self.sparse_imgs[i,:,:,:]
        stop = perf_counter()
        grab_sparse_img = stop - start
        start = perf_counter()
        img = sparse_img.todense().astype(np.float32)
        stop = perf_counter()
        cast_to_dense = stop - start
        img = torch.tensor(img)
        start = perf_counter()
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
        stop = perf_counter()
        rotate_img = stop - start

        ### grab inchi
        start = perf_counter()
        inchi_idx = i + (200000*self.shard_id)
        inchi_data = torch.tensor(self.encoded_inchis[inchi_idx]).long()
        encoded_inchi = inchi_data[:-1]
        inchi_length = inchi_data[-1]
        stop = perf_counter()
        grab_inchi = stop - start
        log_file = open('logs/log_dataloader_times.txt', 'a')
        log_file.write('{},{},{},{}\n'.format(grab_sparse_img, cast_to_dense, rotate_img, grab_inchi))
        log_file.close()
        return img, encoded_inchi, inchi_length

    def __len__(self):
        return self.sparse_imgs.shape[0]
