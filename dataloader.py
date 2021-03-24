import os
from time import perf_counter
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
        open_img_start = perfcounter()
        img_path = get_path_from_img_id(img_id, self.source_dir)
        img = Image.open(img_path)
        open_img_end = perfcounter()
        open_img_time = open_img_end - open_img_start
        convert_img_start = perfcounter()
        img = img.convert('L')
        convert_img_end = perfcounter()
        convert_img_time = convert_img_end - convert_img_start

        rotate_img_start = perf_counter()
        if self.rotate:
            angles = [0, 90, 180, 270]
            angle = np.random.choice(angles, size=1, p=[1 - self.p, self.p / 3, self.p / 3, self.p / 3])
            img = TF.rotate(img, angle, fill=(0,))
        rotate_img_end = perf_counter()
        rotate_img_time = rotate_img_end - rotate_img_start

        invert_img_start = perf_counter()
        img = np.array(img)
        img = invert_and_normalize(img)
        invert_img_end = perf_counter()
        invert_img_time = invert_img_end - invert_img_start

        transform_img_start = perf_counter()
        if self.do_transform:
            img = self.transform(img)
        transform_img_end = perf_counter()
        transform_img_time = transform_img_end - transform_img_start
        img = torch.tensor(img)
        img = img.permute(2, 0, 1).float()

        encode_inchi_start = perf_counter()
        ### grab inchi
        inchi = self.labels.InChI.values[i]
        tokenized_inchi = tokenize_inchi(inchi)
        tokenized_inchi = ['<sos>'] + tokenized_inchi
        tokenized_inchi += ['<eos>']
        inchi_length = torch.tensor(len(tokenized_inchi))
        encoded_inchi = torch.tensor(encode_inchi(tokenized_inchi, self.max_inchi_len, self.char_dict))
        encode_inchi_end = perf_counter()
        encode_inchi_time = encode_inchi_end - encode_inchi_start
        log_file = open('logs/log_dataloader_times.txt', 'a')
        log_file.write('{},{},{},{},{},{},{}\n'.format(i, open_img_time,
                                                       convert_img_time,
                                                       rotate_img_time,
                                                       invert_img_time,
                                                       transform_img_time,
                                                       encode_inchi_time))
        log_file.close()
        return img, encoded_inchi, inchi_length


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

        vertices, window_list = get_vertices(img, window_size = 7,
                                             window_mask = True,
                                             window_list = True)

        closed = closing(prebinarized)

        transformed = np.dstack((img, vertices, closed, edges))
        return transformed
