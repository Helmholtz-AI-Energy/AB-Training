from __future__ import annotations

import glob
import itertools
import logging
import re
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from .utils import rgetattr

log = logging.getLogger(__name__)


@torch.no_grad()
def compare_bases_across_ranks(model: nn.Module, baseline: bool = True):
    """Compare the UVt matrices to each active rank

    Args:
        model (nn.Module): full model to compare
        baselin (bool): is this a traditional model being trained?

    Notes:
        ONLY TO BE USED WITH BASELINE (right now)
    """
    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    ws = dist.get_world_size()

    # need to compare all ranks with all the others and then aggregate the results somehow...
    # also, if some of the basis lines up need to check to see where (since SVD is sorted)
    #   lets just check to see IF they actually line up at all for now
    cossim = nn.CosineSimilarity(dim=0)
    for n, p in model.named_parameters():
        if p.ndim < 2 or not p.requires_grad:
            continue
        # get 2D USV rep -----------------------
        hld = p
        if p.ndim > 2:  # collapse down to 2D
            # shp = p.shape
            hld = p.view(p.shape[0], -1)
        trans = hld.shape[0] < hld.shape[1]
        if trans:  # make 2D rep TS
            hld = hld.T
        u, _, vh = torch.linalg.svd(hld, full_matrices=False)
        uvh = u @ vh
        # --------------------------

        buff = torch.zeros((ws, *list(uvh.shape)), device=u.device, dtype=u.dtype)
        buff[rank] = uvh
        dist.all_reduce(buff)
        if rank == 0:
            log.info(f"Comparins {n}")

        for i, j in itertools.combinations(range(ws), 2):
            sim = cossim(buff[i], buff[j])
            if rank == 0:
                top5 = [f"{tf:.4f}" for tf in sim[:5]]
                if sim.mean() < 0.9:
                    log.info(f"ranks: {i} {j} - Similarity - {sim.mean():.4f}, top 5: {top5}")


@torch.no_grad()
def compare_bases_baseline(model1: nn.Module, model2: nn.Module, baseline: bool = True, print_rank: int = 0):
    """Compare the UVt matrices between two models

    Args:
        model1 (nn.Module): full model to compare
        model2 (nn.Module): full model to compare
        baselin (bool): is this a traditional model being trained?

    Notes:
        ONLY TO BE USED WITH BASELINE (right now)
    """
    # need to compare all ranks with all the others and then aggregate the results somehow...
    # also, if some of the basis lines up need to check to see where (since SVD is sorted)
    #   lets just check to see IF they actually line up at all for now
    cossim = nn.CosineSimilarity(dim=0)
    for n, p in model1.named_parameters():
        if p.ndim < 2 or not p.requires_grad:
            continue
        # get 2D USV rep -----------------------
        hld = p
        if p.ndim > 2:  # collapse down to 2D
            # shp = p.shape
            hld = p.view(p.shape[0], -1)
        trans = hld.shape[0] < hld.shape[1]
        if trans:  # make 2D rep TS
            hld = hld.T
        u, _, vh = torch.linalg.svd(hld, full_matrices=False)
        uvh = u @ vh
        # --------------------------
        # get 2D USV rep model 2 -----------------------
        hld = rgetattr(model2, n)
        if p.ndim > 2:  # collapse down to 2D
            # shp = p.shape
            hld = p.view(p.shape[0], -1)
        trans = hld.shape[0] < hld.shape[1]
        if trans:  # make 2D rep TS
            hld = hld.T
        u2, _, vh2 = torch.linalg.svd(hld, full_matrices=False)
        uvh2 = u2 @ vh2
        # --------------------------
        sim = cossim(uvh, uvh2)
        top5 = [f"{tf:.4f}" for tf in sim[:5]]
        if dist.get_rank() == print_rank:
            if sim.mean() < 0.9:
                log.info(f"Similarity {n[-20:]} - {sim.mean():.4f}, top 5: {top5}")


def get_2d_repr(weight: torch.Tensor) -> list[torch.Tensor, bool, torch.Size]:
    if weight.ndim < 2:
        return None, False, weight.shape
    if weight.ndim == 2:
        trans = weight.shape[0] < weight.shape[1]
        return weight.T if trans else weight, trans, weight.shape

    # if p.ndim > 2:  # collapse down to 2D
    shp = weight.shape
    hld = weight.view(weight.shape[0], -1)
    trans = hld.shape[0] < hld.shape[1]
    return hld.T if trans else hld, trans, shp


def get_og_repr(weight: torch.Tensor, trans: bool, shp: torch.Size, reshape=False, set_loc=None) -> torch.Tensor:
    if trans:
        weight = weight.T
    if shp == weight.shape:
        if set_loc is not None:
            set_loc.zero_()
            set_loc.add_(weight)
            return
        else:
            return weight
    if reshape:
        if set_loc is not None:
            set_loc.zero_()
            set_loc.add_(weight.view(shp))
            return
        else:
            return weight.reshape(shp)


def get_2d_rep_w_1d_names(
    model: nn.Module,
    base_2d: torch.Tensor,
    base_name: str,
    names1d: dict,
) -> tuple[torch.Tensor, dict]:
    """Concatenates 1D tensors to a 2D base tensor along appropriate dimensions.

    This function takes a base 2D tensor and a dictionary containing names of 1D
    tensors. For each 1D tensor specified in `names1d[base_name]`, it identifies the
    matching dimension in the base tensor and concatenates the 1D tensor along the
    other dimension.

    Args:
        model (nn.Module): The PyTorch model containing the 1D weights.
        base_2d (torch.Tensor): The 2D tensor to be expanded.
        base_name (str): Key in the `names1d` dictionary specifying which 1D tensors to use.
        names1d (dict): A dictionary mapping base names to lists of 1D tensor names.

    Returns:
        tuple[torch.Tensor, dict]:
            * The modified 2D tensor with concatenated values.
            * A dictionary mapping the names of the concatenated 1D tensors to their
              corresponding concatenation dimensions.
    """
    # assume that base_2d is a torch tensor that is already 2D, and TS if not square
    cat_dims = {}
    for app_name in names1d[base_name]:
        lp_weight = rgetattr(model, app_name)
        # Find matching dimension in base_2d
        same_dim = torch.nonzero(base_2d.shape == lp_weight.shape).flatten()[0].item()
        other_dim = (same_dim + 1) % 2  # Calculate the other dimension for concatenation
        base_2d = torch.cat([base_2d, lp_weight.unsqueeze(other_dim)], dim=same_dim)
        cat_dims[app_name] = same_dim
    return base_2d, cat_dims


def get_1ds_from_2dcombi(catted2d: torch.Tensor, assoc_1dnames: list, cat_dims: dict) -> tuple[torch.Tensor, dict]:
    """Extracts 1D tensors from a concatenated 2D tensor.

    This function reverses the process of concatenation performed by
    `get_2d_rep_w_1d_names`. It iterates over the associated 1D tensor names
    and uses the concatenation dimensions stored in `cat_dims` to extract the
    original 1D tensors.

    Args:
        catted2d (torch.Tensor): The concatenated 2D tensor.
        assoc_1dnames (list): List of names of the associated 1D tensors.
        cat_dims (dict): Dictionary mapping 1D tensor names to their concatenation dimensions.

    Returns:
        tuple[torch.Tensor, dict]:
            * The remaining portion of the concatenated tensor after extraction.
            * Dictionary containing the extracted 1D tensors, mapped to their names.
    """
    ret = {}
    for n in reversed(assoc_1dnames):  # need to work from the outside in to get everything correctly
        sl = [slice(None), slice(None)]
        sl[cat_dims[n]] = -1
        ret[n] = catted2d[sl]
        sl[cat_dims[n]] = slice(-1)
        catted2d = catted2d[sl]  # remove last dim from this one
    return catted2d, ret


def get_1d_associated_weights(model: torch.Tensor, names_to_ignore: list) -> dict:
    """Get a dictionary which contains the 1D weights which occur after an ND weight
    If the first layers of a network are 1D, they will be ignored.

    The idea is that the layers after the first layer can be appended along the output dim of the layers.
    The input layers are ignored because its unlikely that we can cat along the input dim

    Args:
        model (torch.Tensor): _description_
        names_to_ignore (list): _description_

    Returns:
        dict: A dictionary mapping the names of multidimensional weights to lists
              of associated 1D weight names.
    """
    join_dict = {}
    last_miltidim_w = None

    for n, p in model.named_parameters():
        if n in names_to_ignore:
            continue

        if p.ndim > 1 and last_miltidim_w is not None:
            last_miltidim_w = n
            join_dict[n] = []
        elif last_miltidim_w is not None and p.ndim == 1:
            join_dict[last_miltidim_w].append(n)
    return join_dict
