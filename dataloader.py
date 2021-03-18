import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from util import *

class MoleculeDataset(Dataset):
    """
    PyTorch Dataset class to load molecular images and InChIs
    """
    def __init__(self, labels_fn, source_dir, char_dict, max_inchi_len, do_transform=True, rotate=True, p=0.5):
        self.labels = pd.read_csv(labels_fn)
        self.source_dir = source_dir
        self.char_dict = char_dict
        self.max_inchi_len = max_inchi_len
        self.do_transform = do_transform
        self.rotate = rotate
        self.p = p

    def __getitem__(self, i):
        ### grab image
        img_id = self.labels.image_id.values[i]
        img_path = get_path_from_img_id(img_id, self.source_dir)
        img = Image.open(img_path)
        img.convert('L')

        if self.rotate:
            angles = [0, 90, 180, 270]
            angle = np.random.choice(angles, size=1, p=[1 - self.p, self.p / 3, self.p / 3, self.p / 3])
            img = TF.rotate(img, angle)

        img = np.array(img)
        img = invert_and_normalize(img)

        if self.do_transform:
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

    def transform(self, img):
        """
        Takes in a 2D grayscale image and turns into multi-channel array. Each channel
        stores a different type of transformation.

        Currently, channels are: [img, vertices, dilated, eroded,
                                  enhanced and detected edges]
        """
        prebinarized = binarize(img)

        edges = edge_enhance(prebinarized)
        edges = edge_detect(edges)

    #     vertices = get_vertices(img, window_size = 5, window_mask = True)
        vertices = get_vertices(img, window_size = 3, window_mask = False)

        dilated = dilate(img)
        eroded = erode(dilated)

        transformed = np.dstack((img, vertices, dilated, eroded, edges))
        return transformed
