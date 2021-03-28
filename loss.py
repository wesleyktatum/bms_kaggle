import torch
import torch.nn
import torch.nn.functional as F

def ce_loss(targets, preds, alphas, weights, alpha_c):
    BCE = F.cross_entropy(preds, targets, reduction='mean', weight=weights)
    ALPHA_NORM = alpha_c * ((1. - alphas.sum(dim=1))**2).mean()
    return BCE + ALPHA_NORM, BCE, ALPHA_NORM
