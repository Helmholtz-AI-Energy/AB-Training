from __future__ import annotations

import glob
import logging
import re
import time
from functools import reduce, wraps
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

log = logging.getLogger(__name__)


def only_on_rank_n(run_on_rank: int = 0):
    # run a command only on a specified rank
    # this is a decorator: i.e.: @only_on_rank_n("rank1") will run only on rank 1

    # three options: run on all, run on 0, run on target
    rank = 0  # start with run on all - stick with this if not distributed
    target_rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
        target_rank = run_on_rank
        if run_on_rank < 0:  # run on all
            target_rank = rank

    def rank_zero_only(fn: Callable) -> Callable:
        """Wrap a function to call internal function only in rank zero.
        Function that can be used as a decorator to enable a function/method
        being called only on global rank 0.
        """

        @wraps(fn)
        def wrapped_fn(*args: Any, **kwargs: Any) -> Any | None:
            # rank = getattr(rank_zero_only, "rank", None)
            # if rank is None:
            #     raise RuntimeError("torch distributed not initialized yet")
            if rank == target_rank:
                return fn(*args, **kwargs)
            return None

        return wrapped_fn

    return rank_zero_only


def print0(*args, sep=" ", end="\n", file=None):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(*args, sep=sep, end=end, file=file)
    elif dist.is_initialized():
        return
    else:
        print(*args, sep=sep, end=end, file=file)


@only_on_rank_n(0)
def log0(msg, level, logger=None, *args, **kwargs):
    if logger is None:
        logger = logging.getLogger(__name__)
    getattr(logger, level)(msg, *args, **kwargs)


def change_batchnorm_tracking(model: nn.Module, tracking=False):
    for child in model.children():
        if hasattr(child, "track_running_stats"):
            child.track_running_stats = tracking
            # child.training = tracking
            # child.affine = tracking
        change_batchnorm_tracking(child, tracking)


def increment_path(path, exist_ok=True, sep=""):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    base_path = Path(path)

    counter = 0
    path = base_path / str(counter)
    while path.exists():
        counter += 1
        path = base_path / str(counter)
    return path


def roll_orthogonal_values(
    shape: Union[Tuple, List, torch.Size],
    dtype: torch.dtype,
    device: torch.device,
    scale_factor: float = 0.1,
):
    """
    References
    ----------
    .. [1] F. Mezzadri, "How to generate random matrices from the classical
       compact groups", :arXiv:`math-ph/0609050v2`.

    Parameters
    ----------
    shape
    dtype
    device
    scale_factor

    Returns
    -------

    """
    # roll new "gradients" to search the local area
    # the gradients are orthogonal matrices, normally between -1 and 1
    # TODO: since the parameters are typically on the same order of magnitude, it may be
    #  needed to scale the gradients. TBD.
    if len(shape) == 1:
        return torch.randn(shape, dtype=dtype, device=device) * scale_factor

    z = torch.randn(shape, dtype=dtype, device=device)
    if shape[-1] > shape[-2]:
        # need to transpose? or just do complete, then slice of the bad bits
        if len(shape) > 2:
            hold = torch.arange(len(shape) - 2)
            x_perm = z.permute(*hold, -1, -2)
            q, r = torch.linalg.qr(x_perm, mode="reduced")
            d = r.diagonal()
            ret = q * (d / d.abs()).unsqueeze(-2)
            ret = ret[..., : x_perm.shape[-1]]
            ret = ret.permute(*hold, -1, -2)
        else:
            x_perm = z.permute(-1, -2)
            q, r = torch.linalg.qr(x_perm, mode="reduced")
            d = r.diagonal()
            # print('h', q.shape, d.shape)
            ret = q * (d / d.abs()).unsqueeze(-2)
            ret = ret[..., : x_perm.shape[-1]]
            ret = ret.permute(-1, -2)
    else:
        z = torch.randn(shape, dtype=dtype, device=device)
        q, r = torch.linalg.qr(z, mode="reduced")
        d = r.diagonal()
        ret = q * (d / d.abs()).unsqueeze(-2)
    return ret * scale_factor


def reroll_model(layer: nn.Module):
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
    else:
        if hasattr(layer, "children"):
            for child in layer.children():
                reroll_model(child)


@torch.no_grad()
def modify_model_random_simgas(model: nn.Module, device: torch.device, mode: str = "rand"):
    """
    objective: roll the sigma matrix of the SVD representation of the multi-dim weights to be random
               but the bases (U/V) of the reps will be the same on each rank

    steps:
        1. if the seed everywhere is the same for torch: move to sigma step
           else: fix seed + re-init model
        2. sigma step - create generator, roll random vals, scale to match original, sort?

    Args:
        model (nn.Module): model to work on
        mode (str): mode for modifying sigmas
            'rand' - sigmas are rolled randomly everywhere
            'ortho' - sigmas are set to one
    """
    if mode not in ["rand", "ortho"]:
        raise ValueError(f"mode arg must be in {['rand', 'ortho']}, currently: {mode}")
    if not dist.is_initialized():
        return
    log.info("Rolling rankdom sigma values for multi dim weights")
    # check if seeds are equal
    rank = dist.get_rank()
    ws = dist.get_world_size()
    loc_seed = torch.random.initial_seed()
    seeds = torch.zeros(ws, device=device)  # todo: fix device?
    seeds[rank] = loc_seed
    dist.all_reduce(seeds)
    if not torch.all(seeds == loc_seed):
        # there exist other seeds (models are not the same), need to unify and roll weights again
        torch.manual_seed(seeds[0])
        reroll_model(model)
    # seeds and models are all the same now
    # next: create a unique generator on each rank and roll sigma values
    gen = torch.Generator(device=device)
    gen = gen.manual_seed(int(seeds[0].item()) + rank)
    for _n, p in model.named_parameters():
        if p.ndim < 2:  # skip params with < 2D
            continue
        hld = p
        if p.ndim > 2:  # collapse down to 2D
            shp = p.shape
            hld = p.view(p.shape[0], -1)
        trans = hld.shape[0] < hld.shape[1]
        if trans:  # make 2D rep TS
            hld = hld.T
        u, s, vh = torch.linalg.svd(hld, full_matrices=False)
        if mode == "ortho":
            news = torch.ones_like(s)
        else:
            news = torch.rand(s.shape[0], device=s.device, dtype=s.dtype, generator=gen)
            news *= s[0]
        # NOTE: this will ALWAYS shrink some of the values, range is [0, 1)
        hld = u @ torch.diag(news) @ vh
        if trans:
            hld = hld.T
        if p.ndim > 2:
            hld = hld.view(shp)
        # need to have contiguous for future torch internals
        p.zero_()
        p.add_(hld)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


def reset_adam_state(optimizer, p):
    """
    reset the shapes of the Adam optimizer buffers to be the same shape as the model parameters

    if `reset_buffers_zero`: reset the buffer to zero after reshaping it
    """
    # resettime = time.perf_counter()
    # rank = 0 if not dist.is_initialized() else dist.get_rank()
    # instead of resetting optimizer, slice off bits of the saved states

    for group in optimizer.param_groups:
        state = optimizer.state[p]
        if len(list(state.keys())) > 0:
            for k in ["exp_avg", "exp_avg_sq"]:
                state[k] *= 0
            if group["amsgrad"]:
                state["max_exp_avg_sq"] *= 0
    # if rank == 0:
    #     log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")
