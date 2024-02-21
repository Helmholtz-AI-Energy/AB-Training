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
