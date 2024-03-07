import itertools
import logging

import torch
import torch.distributed as dist
import torch.nn as nn

from .layers import ABConv, ABLinear

log = logging.getLogger(__name__)


def convert_network_ab_lowrank(module, config=None, skip_modules=None):
    module_output = module
    if skip_modules is None:
        skip_modules = []

    if hasattr(module, "weight") and module.weight.squeeze().ndim > 1 and module not in skip_modules:
        if isinstance(module, nn.Linear):
            module_output = ABLinear(module, config)
            module_output = module_output.to(device=module.weight.device, dtype=module.weight.dtype)
        elif isinstance(module, nn.modules.conv._ConvNd):
            module_output = ABConv(module, config)
            module_output = module_output.to(device=module.weight.device, dtype=module.weight.dtype)
        else:
            log.info(f"NEW LAYER TO WRAP: {type(module)} weight shape: {module.weight.shape}")

    for name, child in module.named_children():
        module_output.add_module(
            name,
            convert_network_ab_lowrank(child, config, skip_modules),
        )
    del module
    return module_output


def change_ab_train_mode(module, ab_training_mode, cut_singular_values):
    if hasattr(module, "change_ab_train_mode"):
        module.change_ab_train_mode(ab_training_mode, cut_singular_values)

    for _name, child in module.named_children():
        change_ab_train_mode(child, ab_training_mode, cut_singular_values)


def compare_basis_ab(module, name=None):
    if hasattr(module, "compare_bases"):
        module.compare_bases(name)

    for name, child in module.named_children():
        compare_basis_ab(child, name)


def _get_network_compression(
    module: nn.Module,
    currently_trainable=0,
    total_current_params=0,
    full_rank_param_count=0,
):
    if len(list(module.children())) == 0:
        if not hasattr(module, "get_compression"):
            for p in module.parameters():
                if p.requires_grad:
                    currently_trainable += p.numel()
                total_current_params += p.numel()
                full_rank_param_count += p.numel()
        else:
            trainable, total, full_rank = module.get_compression()
            currently_trainable += trainable
            total_current_params += total
            full_rank_param_count += full_rank
    else:
        for n, c in module.named_children():
            currently_trainable, total_current_params, full_rank_param_count = _get_network_compression(
                c,
                currently_trainable,
                total_current_params,
                full_rank_param_count,
            )
    # print(currently_trainable, total_current_params, full_rank_param_count)
    return currently_trainable, total_current_params, full_rank_param_count


def get_network_compression(module):
    currently_trainable, total_current_params, full_rank_param_count = _get_network_compression(module)
    # print(currently_trainable, total_current_params, full_rank_param_count)
    if dist.get_rank() == 0:
        log.info(
            f"Network Compression: Trainable params: {currently_trainable}, "
            f"total current params {total_current_params} "
            f"full rank param count {full_rank_param_count}\t"
            f"Compression: {(total_current_params / full_rank_param_count) * 100:.3f}",
        )
