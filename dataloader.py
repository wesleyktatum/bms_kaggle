import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from util import *

class MoleculeDataset(Dataset):
    """
    PyTorch Dataset class to load molecular images and InChIs
    """
    def __init__(self, labels_fn, source_dir, char_dict, max_inchi_len, transform=None):
        self.labels = pd.read_csv(labels_fn)
        self.source_dir = source_dir
        self.char_dict = char_dict
        self.max_inchi_len = max_inchi_len
        self.transform = transform

    def __getitem__(self, i):
        ### grab image
        img_id = self.labels.image_id.values[i]
        img_path = get_path_from_img_id(img_id, self.source_dir)
        img = (255 - cv2.imread(img_path)) / 255
        img = img[:,:,0]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.tensor(img)

        ### grab inchi
        inchi = self.labels.InChI.values[i]
        inchi = inchi.split('InChI=1S/')[1]
        inchi = ''.join(inchi)
        tokenized_inchi = tokenize_inchi(inchi)
        tokenized_inchi = ['<sos>'] + tokenized_inchi
        tokenized_inchi += ['<eos>']
        encoded_inchi = torch.tensor(encode_inchi(tokenized_inchi, self.max_inchi_len, self.char_dict))
        return img, encoded_inchi


    def __len__(self):
        return self.labels.shape[0]
