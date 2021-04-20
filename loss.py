import torch
import torch.nn
import torch.nn.functional as F

def ce_loss(targets, preds, weights):
    BCE = F.cross_entropy(preds, targets, reduction='mean', weight=weights)
    return BCE

def focal_loss(targets, preds, weights, gamma=2):
    ce_loss = F.cross_entropy(preds, targets, reduction='none', weight=weights)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt)**gamma * ce_loss).mean()
    return focal_loss
