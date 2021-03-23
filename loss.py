import torch
import torch.nn
import torch.nn.functional as F

def ce_loss(targets, preds, weights):
    BCE = F.cross_entropy(preds, targets, reduction='mean', weight=weights)
    return BCE
