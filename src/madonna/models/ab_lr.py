import logging

import torch
import torch.distributed as dist
import torch.nn as nn

from ..utils import basis

log = logging.getLogger(__name__)


class ABLowRankLayer(nn.Module):
    def __init__(self, base_layer, config=None):
        # class objective: replace the weight with an AB low-rank decomp
        super().__init__()
        self.base_layer = base_layer
        self.full_rank_weight = base_layer.weight
        self.training_mode_ab = "full"
        self.a = nn.Parameter(torch.empty(1), requires_grad=False)
        self.b = nn.Parameter(torch.empty(1), requires_grad=False)
        if config is not None:
            self.config = config
            self.low_rank_cutoff = config.training.ab.low_rank_cutoff
            self.split_sigma = config.training.ab.split_sigma
            self.logging_rank = config.training.logging_rank
        else:
            self.config = None
            self.low_rank_cutoff = 0.1
            self.split_sigma = True
            self.logging_rank = 0
        self.trans = None
        self.shp = self.full_rank_weight.shape
        self.full_numel = self.full_rank_weight.numel()

    @torch.no_grad()
    def change_ab_train_mode(self, mode, cut_vals: bool):
        """In here, need to change between training modes
        If we were in full, need to set a and b based on the SVD and do cutoff of values
        If we were in A/B, need to set full rank weights

        Args:
            mode (_type_): _description_
        """
        if mode not in ["full", "a", "b"]:
            raise ValueError(f"mode must be in ['full', 'a', 'b']: current: {mode}")
        if self.training_mode_ab == mode:
            return
        self.training_mode_ab = mode
        if mode == "full":
            # todo: set up normal training here if not first pass
            if self.a.ndim > 1:
                self.target_weight = self.a @ self.b
            self._set_req_grad_flags(True)
            return
        self.set_ab(mode=mode, cut_vals=cut_vals)
        self._set_req_grad_flags(True)

    @torch.no_grad()
    def set_ab(self, mode, cut_vals=False):
        """set A and B based off the mode of training
        If mode == a -> A = U @ sigma.diag() and B = Vh
        if mode == b -> A = U and B = sigma.diag() @ Vh

        Args:
            mode (_type_): _description_
        """
        twodrepr, trans, _ = basis.get_2d_repr(self.full_rank_weight)
        if self.trans is None:
            self.trans = trans
        u, s, vh = torch.linalg.svd(twodrepr, full_matrices=False)
        if cut_vals:
            # TODO: logging
            cut = torch.nonzero(s > s[0] * self.low_rank_cutoff).flatten()[-1]
            u = u[:, : cut + 1].clone()
            s = s[: cut + 1].clone()
            vh = vh[: cut + 1].clone()
            compression = ((s.shape[0] * (u.shape[0] + vh.shape[1])) / self.full_numel) * 100
            log.info(
                f"Curring shape of orig: {tuple(self.shp)}, ({u.shape[0]}, {s.shape[0]}, {vh.shape[1]}), "
                f"comp: {compression}",
            )

        if self.split_sigma:
            self.a = u @ s.sqrt().diag()
            self.b = s.sqrt().diag() @ vh
        elif mode == "a":
            self.a = u @ s.diag()
            self.b = vh
        else:  # if mode == "b": -> only other option....hopefully
            self.a = u
            self.b = s.diag() @ vh

    def _set_req_grad_flags(self, mode=True):
        if self.training_mode_ab == "full":
            self.target_weight.requires_grad = mode
            self.base_layer.weight.requires_grad = mode
            self.a.requires_grad = False
            self.b.requires_grad = False
        elif self.training_mode_ab == "a":
            self.target_weight.requires_grad = False
            self.base_layer.weight.requires_grad = False
            self.a.requires_grad = mode
            self.b.requires_grad = False
        elif self.training_mode_ab == "b":
            self.target_weight.requires_grad = False
            self.base_layer.weight.requires_grad = False
            self.a.requires_grad = False
            self.b.requires_grad = mode

    def train(self, mode):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.base_layer.train(mode)
        # call on internal layers, then set the weights and such to be what we want
        self._set_req_grad_flags(mode)

    def set_weight(self):
        if self.training_mode_ab == "full":
            self.base_layer.weight = self.target_weight
        else:
            weight = self.a @ self.b
            weight = weight.T if self.trans else weight
            # TODO: calling view might be an issue for backward here, might need contiguous
            self.base_layer.weight = weight.view(self.shp)

    def forward(self, *input, **kwargs):
        self.set_weight()
        return self.base_layer(*input, **kwargs)


def convert_network_ab_lowrank(module, config=None):
    module_output = module
    if hasattr(module, "weight") and module.weight.squeeze().ndim > 1:
        module_output = ABLowRankLayer(module, config)

    for name, child in module.named_children():
        module_output.add_module(
            name,
            convert_network_ab_lowrank(child, config),
        )
    del module
    return module_output


def change_ab_train_mode(module, ab_training_mode, cut_singular_values):
    if hasattr(module, "change_ab_train_mode"):
        module.change_ab_train_mode(ab_training_mode, cut_singular_values)

    for _name, child in module.named_children():
        change_ab_train_mode(child, ab_training_mode, cut_singular_values)
