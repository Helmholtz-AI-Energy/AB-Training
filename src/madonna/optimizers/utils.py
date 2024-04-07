import logging
import time
from typing import List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

# from mpi4py import MPI

# from ..utils import comm, utils

log = logging.getLogger(__name__)


# need function to reset the sizes of the optimizer buffers


def change_adam_shapes(optimizer):
    """
    reset the shapes of the Adam optimizer buffers to be the same shape as the model parameters

    if `reset_buffers_zero`: reset the buffer to zero after reshaping it
    """
    resettime = time.perf_counter()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    # instead of resetting optimizer, slice off bits of the saved states
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if len(list(state.keys())) > 0:
                for k in ["exp_avg", "exp_avg_sq"]:
                    if state[k].shape != p.shape:
                        # new_state = truncate_tensor(state[k], p.shape)
                        new_state, pading, slices = pad_or_truncate_tensor(
                            input_tensor=state[k],
                            target_shape=p.shape,
                            padding_value=0,
                        )
                        state[k] = new_state.contiguous()
                if "amsgrad" in group and group["amsgrad"]:
                    if state["max_exp_avg_sq"].shape != p.shape:
                        k = "max_exp_avg_sq"
                        new_state = pad_or_truncate_tensor(
                            input_tensor=state[k],
                            target_shape=p.shape,
                            padding_value=0,
                        )
                        state[k] = new_state
    if rank == 0:
        log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")


def change_sgd_shapes(optimizer, model, reset_buffers_zero: bool = False, param_indices=None):
    """
    reset the shapes of the SGD optimizer buffers to be the same shape as the model parameters

    if `reset_buffers_zero`: reset the buffer to zero after reshaping it
    """
    resettime = time.perf_counter()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    # print(list(optimizer.param_groups[0].keys()))
    # raise ValueError
    for c, (n, p) in enumerate(model.named_parameters()):
        if param_indices is not None and c not in param_indices:
            continue
        # if dist.get_rank() == 0:
        #     print(n, optimizer.param_groups[0]["params"][c].shape, p.shape)
        settozero = False
        if n.endswith((".s", "_s")) and p.requires_grad and reset_buffers_zero:
            settozero = True

        state = optimizer.state[p]
        # print(list(state.keys()))
        if len(list(state.keys())) > 0:
            # if len(optimizer.param_groups[0]["momentum_buffer"]) > 0:
            if state["momentum_buffer"].shape != p.shape:
                sl = []
                for d in range(p.ndim):
                    sl.append(slice(0, p.shape[d]))
                state["momentum_buffer"] = state["momentum_buffer"][tuple(sl)]
            if settozero:
                state["momentum_buffer"].zero_()

        # if optimizer.param_groups[0]["params"][c].shape != p.shape:
        #     sl = []
        #     for d in range(p.ndim):
        #         sl.append(slice(0, p.shape[d]))
        #     optimizer.param_groups[0]["params"][c] = optimizer.param_groups[0]["params"][c][tuple(sl)]
        #     optimizer.param_groups[0]["params"][c].zero_()
    if rank == 0:
        log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")


def pad_or_truncate_tensor(input_tensor, target_shape, padding_value=0):
    """Pads or truncates a tensor to a given shape.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        target_shape (tuple): The desired shape of the output tensor.
        padding_value (int or float): The value to use for padding (default: 0).

    Returns:
        torch.Tensor: The padded or truncated tensor.
    """
    if tuple(input_tensor.shape) == tuple(target_shape):
        return input_tensor
    pad_list = []
    slices = []
    # pad list is weird in torch. it works from the outside in
    # that means each padding the look has to be reversed
    for current_dim, target_dim in zip(reversed(input_tensor.shape), reversed(target_shape)):
        if target_dim > current_dim:
            pad_list.extend([0, target_dim - current_dim])  # Pad on the right side
            slices.append(slice(target_dim))
        elif target_dim < current_dim:
            pad_list.extend([0, 0])  # No padding
            slices.append(slice(target_dim))
        else:
            pad_list.extend([0, 0])  # No padding
            slices.append(slice(target_dim))

    slices = [sl for sl in reversed(slices)]

    output_tensor = torch.nn.functional.pad(input_tensor, pad_list, value=padding_value)
    return output_tensor[slices], pad_list, slices


def truncate_tensor(input_tensor, target_shape):
    """Truncates a tensor to a given shape.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        target_shape (tuple): The desired shape of the output tensor.

    Returns:
        torch.Tensor: The truncated tensor.
    """
    if tuple(input_tensor.shape) == tuple(target_shape):
        return input_tensor
    slices = [slice(0, dim) for dim in target_shape]
    return input_tensor[slices]
