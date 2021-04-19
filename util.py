import os
import re
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from skimage import feature, filters, morphology
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.sasa import ResNet26, ResNet38, ResNet50
from models.axial import axial18s, axial18srpe, axial26s, axial50s, axial50m, axial50l
from models.resnet import resnet18, resnet34, resnet50
from models.bilstm import biLSTM512
from models.transformer import trans128_4x, trans256_4x, trans512_4x
from models.caption import CaptionModel

########################################################
############# PROCESSING FUNCTIONS #####################
########################################################

def get_path_from_img_id(img_id, DIR):
    img_path = os.path.join(DIR, img_id[0], img_id[1], img_id[2], '{}.png'.format(img_id))
    return img_path

def get_n_shards(dir):
    pattern = "[0-9]"
    regezz = re.compile(pattern)
    shard_ids = []
    for fn in os.listdir(dir):
        nums = [num for num in regezz.findall(fn)]
        if len(nums) > 0:
            shard_ids.append(int(nums[0]))
    n_shards = max(shard_ids) + 1
    return n_shards

def calc_data_mean_std(img_ids, DIR):
    channel_1_means = []
    channel_2_means = []
    channel_3_means = []
    channel_1_stds = []
    channel_2_stds = []
    channel_3_stds = []
    for img_id in img_ids:
        img_path = get_path_from_img_id(img_id, DIR)
        img = Image.open(img_path)
        img = img.convert('L')
        img = np.array(img)
        img = invert_and_normalize(img)
        means = np.mean(img.reshape(-1, img.shape[-1]), axis=0)
        stds = np.std(img.reshape(-1, img.shape[-1]), axis=0)
        channel_1_means.append(means[0])
        channel_2_means.append(means[1])
        channel_3_means.append(means[2])
        channel_1_stds.append(stds[0])
        channel_2_stds.append(stds[1])
        channel_3_stds.append(stds[2])
    channel_1_mean = np.mean(channel_1_means)
    channel_2_mean = np.mean(channel_2_means)
    channel_3_mean = np.mean(channel_3_means)
    channel_1_std = np.mean(channel_1_stds)
    channel_2_std = np.mean(channel_2_stds)
    channel_3_std = np.mean(channel_3_stds)
    means = [channel_1_mean, channel_2_mean, channel_3_mean]
    stds = [channel_1_std, channel_2_std, channel_3_std]
    mean_std_dict = {'means': means,
                     'stds': stds}
    with open(os.path.join(DIR, 'mean_std.json'), 'w') as f:
        json.dump(mean_std_dict, f)

def tokenize_inchi(inchi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|Si?|N|H|O|S|P|F|I|D|T|b|c|n|o|s|p|h|t|m|i|\(|\)|\.|=|#|-|,|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|1[0-9]|2[0-9]|[0-9])"
    regezz = re.compile(pattern)
    inchi = inchi.split('InChI=1S/')[1]
    inchi = ''.join(inchi)
    tokens = [token for token in regezz.findall(inchi)]
    assert inchi == ''.join(tokens), ("{} could not be joined -> {}".format(inchi, tokens))
    return tokens

def encode_inchi(inchi, max_len, char_dict):
    "Converts tokenized InChIs to a list of token ids"
    for i in range(max_len - len(inchi)):
        inchi.append('<pad>')
    inchi_vec = [char_dict[c] for c in inchi]
    return inchi_vec

def decode_inchi(preds, ord_dict, probs=False):
    if probs:
        preds = F.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1).squeeze(0).numpy()
    else:
        preds = preds.numpy()
    inchi = 'InChI=1S/'
    for idx in preds:
        tok = ord_dict[str(idx)]
        if tok == '<eos>':
            return inchi
        inchi += tok
    return inchi

def get_char_weights(train_inchis, params, freq_penalty=0.5):
    "Calculates token weights for a set of input data"
    char_dist = {}
    char_counts = np.zeros((params['NUM_CHAR'],))
    char_weights = np.zeros((params['NUM_CHAR'],))
    for k in params['CHAR_DICT'].keys():
        char_dist[k] = 0
    for inchi in train_inchis:
        for i, char in enumerate(inchi):
            char_dist[char] += 1
        for j in range(i, params['MAX_LENGTH']):
            char_dist['<pad>'] += 1
    for i, v in enumerate(char_dist.values()):
        char_counts[i] = v
    top = np.sum(np.log(char_counts))
    for i in range(char_counts.shape[0]):
        char_weights[i] = top / np.log(char_counts[i])
    min_weight = char_weights.min()
    for i, w in enumerate(char_weights):
        if w > 2*min_weight:
            char_weights[i] = 2*min_weight
    scaler = MinMaxScaler([freq_penalty,1.0])
    char_weights = scaler.fit_transform(char_weights.reshape(-1, 1))
    return char_weights[:,0]

########################################################
################## MODEL FUNCTIONS #####################
########################################################

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

def load_model_from_ckpt(ckpt_fn):
    ckpt = torch.load(ckpt_fn, map_location=torch.device('cpu'))
    args = ckpt['args']
    start_epoch = ckpt['epoch']
    return ckpt, args, start_epoch

class MixScheduler:
    def __init__(self, alpha_init=1., alpha_fin=0.1, step_rate=1e-3, warmup_steps=35000):
        self.alpha = alpha_init
        self.alpha_init = alpha_init
        self.alpha_fin = alpha_fin
        self.step_rate = step_rate
        self.n_steps = 0
        self.warmup_steps = warmup_steps

    def step(self):
        if self.n_steps < self.warmup_steps:
            self.n_steps += 1
        else:
            self.alpha -= ((1. - self.alpha_fin) * (self.step_rate / 100))
            if self.alpha <= self.alpha_fin:
                self.alpha = self.alpha_fin
            self.n_steps += 1

    def get_alpha(self):
        return self.alpha

########################################################
############# IMAGE TRANSFORM FUNCTIONS ################
########################################################

def invert_and_normalize(img):
    new_im = (255 - img) / 255
    return new_im


def binarize(img):
#     thresh = filters.threshold_yen(img)
    thresh = 0.3
    threshed = img > thresh
    binary = np.where(threshed == True, 1, 0)
    return binary


def dilate(img):
    dilated = morphology.dilation(img, selem = morphology.octagon(1, 1))
    return dilated


def erode(img):
    eroded = morphology.erosion(img, selem = morphology.octagon(1, 1))
    return eroded


def opening(img):
    opened = morphology.opening(img, selem = morphology.star(1))
    return opened


def closing(img):
    closed = morphology.closing(img, selem = morphology.octagon(1, 1))
    return closed


def edge_enhance(img):
    sharpened = filters.unsharp_mask(img, radius = 4, amount = 3)
    return sharpened


def edge_detect(img):
#     edges = feature.canny(img)
    edges = filters.sobel(img)
    return edges


def get_window_mask(x, y, vertex_map, window_size = 5):
    n_rows = (window_size * 2) + 1
    if x - window_size < 0:
        x_start = 0
    else:
        x_start = x - window_size
    if x + window_size >= 256:
        x_stop = 255
    else:
        x_stop = x + window_size
    for i in range(n_rows):
        if i <= window_size-1:
            row = y - (window_size - i)
            if row < 0:
                row = 0
            if row >= 256:
                row = 255
            vertex_map[x_start:x_stop, row] = 1

        if i == (window_size + 1):
            row = y
            if row < 0:
                row = 0
            if row >= 256:
                row = 255
            vertex_map[x_start:x_stop, row] = 1

        if i >= (window_size + 1):
            row = y + (i - window_size)
            if row < 0:
                row = 0
            if row >= 256:
                row = 255
            vertex_map[x_start:x_stop, row] = 1

    return vertex_map


def get_img_window(x, y, img, window_size = 5):
    window = img[(x-window_size):(x+window_size), (y-window_size):y+window_size]
    return window


def get_window_coords(img):
    coords = feature.corner_peaks(feature.corner_harris(img), min_distance=5,
                                  threshold_rel=0.02)
    return coords


def get_vertex_map(coords, img, window_size, window_list):
    h, w = img.shape
    vertex_map = np.zeros((h, w))
    windows = {'coordinates':[],
               'imgs': []}

    for i, coordinate in enumerate(coords):
        x, y = coordinate

        if window_size > 0:
                vertex_map = get_window_mask(x, y, vertex_map,
                                             window_size = window_size)
        else:
            vertex_map[x, y] = 1

        if window_list:
            window = get_img_window(x, y, img, window_size = window_size)
            windows['imgs'].append(window)
            windows['coordinates'].append((x, y))

    if window_list:
        return vertex_map, windows
    else:
        return vertex_map, None


def get_vertices(img, window_size = 7, window_mask = True, window_list = True):

    coords = get_window_coords(img)

    vertex_map, windows = get_vertex_map(coords, img, window_size = window_size,
                                         window_list = window_list)

    if window_mask:
        if window_list:
            vertex_windows = np.where(vertex_map == 1, img, 0)
            return vertex_windows, windows

        else:
            vertex_windows = np.where(vertex_map == 1, img, 0)
            return vertex_windows, None
    else:
        return vertex_map, None


def morph_around_windows(img, windows, morph_function):
    """
    Ensure that window_list only contains windows that are to be avoided during
    morphological operations (e.g. closing). window mask is created and morph_function
    is applied to resulting image
    """

    vertex_mask, _ = get_vertex_map(windows['coordinates'], img, window_size = 5,
                                    window_list = False)

    morphed_img = morph_function(img)

    final_img = np.where(vertex_mask == 1, img, morphed_img)

    return final_img

def preprocess(img_path, extensive=False):
    """
    Takes in a path to a 2D grayscale image and turns into multi-channel array.
    Each channel stores a different type of transformation.

    Currently, channels are: [img, vertices]
    """
    img = Image.open(img_path)
    img = img.convert('L')
    img = np.array(img)
    img = invert_and_normalize(img)
    vertices, window_list = get_vertices(img, window_size=5, window_mask=False,
                                         window_list=True)

    if extensive:
        prebinarized = binarize(img)
        edges = edge_enhance(prebinarized)
        edges = edge_detect(edges)
        closed = closing(prebinarized)
        img = np.dstack((img, vertices, edges, closed))
    else:
        img = np.dstack((img, vertices))
    img = np.transpose(img, (2, 0, 1))
    return img


########################################################
######## TROUBLESHOOTING FUNCTIONS #####################
########################################################

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n.split('.weight')[0])
            ave_grads.append(p.grad.abs().mean())
    layers = np.array(layers)
    ave_grads = np.array(ave_grads)
    fig = plt.figure(figsize=(14,6))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=10)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.ylim(ymin=0, ymax=5e-3)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.grid(True)
    plt.tight_layout()
    return plt
