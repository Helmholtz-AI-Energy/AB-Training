import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def mask_prune_model(module, precomputed_mask):
    prune.custom_from_mask(module, "weight", mask=precomputed_mask)
