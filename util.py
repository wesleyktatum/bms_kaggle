import os
import re
import json
import numpy as np

def get_path_from_img_id(img_id, DIR):
    img_path = os.path.join(DIR, img_id[0], img_id[1], img_id[2], '{}.png'.format(img_id))
    return img_path

def calc_data_mean_std(img_ids, DIR):
    channel_1_means = []
    channel_2_means = []
    channel_3_means = []
    channel_1_stds = []
    channel_2_stds = []
    channel_3_stds = []
    for img_id in img_ids:
        img_path = get_path_from_img_id(img_id, DIR)
        img = (255 - cv2.imread(img_path)) / 255
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
    tokens = [token for token in regezz.findall(inchi)]
    assert inchi == ''.join(tokens), ("{} could not be joined -> {}".format(inchi, tokens))
    return tokens

def encode_inchi(inchi, max_len, char_dict):
    "Converts tokenized InChIs to a list of token ids"
    for i in range(max_len - len(inchi)):
        inchi.append('<pad>')
    inchi_vec = [char_dict[c] for c in inchi]
    return inchi_vec
