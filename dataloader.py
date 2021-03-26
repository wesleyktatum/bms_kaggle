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
    def __init__(self, labels_fn, source_dir, shard_id, char_dict,
                 max_inchi_len, rotate=True, p=0.5):
        labels = pd.read_csv(labels_fn)
        self.inchis = labels.InChI.values
        self.mode = labels_fn.split('/')[-1].split('.')[0]
        self.shard_id = shard_id
        self.sparse_path = os.path.join(source_dir, '{}_shards'.format(self.mode), 'shard{}.npz'.format(shard_id))
        self.sparse_imgs = sparse.load_npz(self.sparse_path)
        self.char_dict = char_dict
        self.max_inchi_len = max_inchi_len
        self.rotate = rotate
        self.p = p

    def __getitem__(self, i):
        ### grab image
        sparse_img = self.sparse_imgs[i,:,:,:]
        img = sparse_img.todense()
        img = torch.tensor(img)
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

        ### grab inchi
        inchi_idx = i + (200000*self.shard_id)
        inchi = self.inchis[inchi_idx]
        tokenized_inchi = tokenize_inchi(inchi)
        tokenized_inchi = ['<sos>'] + tokenized_inchi
        tokenized_inchi += ['<eos>']
        inchi_length = torch.tensor(len(tokenized_inchi))
        encoded_inchi = torch.tensor(encode_inchi(tokenized_inchi, self.max_inchi_len, self.char_dict))
        return img, encoded_inchi, inchi_length

    def __len__(self):
        return self.sparse_imgs.shape[0]
