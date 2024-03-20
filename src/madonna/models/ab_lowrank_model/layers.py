import itertools
import logging
from copy import deepcopy
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ...utils import basis

log = logging.getLogger(__name__)


class ABLowRankLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, config: dict = None, mutli_dim_bias: bool = False):
        # class objective: replace the weight with an AB low-rank decomp
        # TODO: should super be called or not??
        # super().__init__()
        self.full_rank_weight = nn.Parameter(base_layer.weight.clone().detach(), requires_grad=False)
        self.multi_dim_bias = mutli_dim_bias
        if mutli_dim_bias:
            if base_layer.bias.ndim > 1:
                self.full_rank_bias = nn.Parameter(base_layer.bias.clone().detach(), requires_grad=False)
                self.a_bias = nn.Parameter(torch.empty(1), requires_grad=True)
                self.b_bias = nn.Parameter(torch.empty(1), requires_grad=True)
                self.bias_shp = self.base_layer.bias.shape
                self.bias_trans = None
            else:
                self.multi_dim_bias = False

        self.training_mode_ab = "full"
        self.a = nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = nn.Parameter(torch.empty(1), requires_grad=True)
        if config is not None:
            self.config = config
            self.low_rank_cutoff = config["training"]["ab"]["low_rank_cutoff"]
            self.split_sigma = config["training"]["ab"]["split_sigma"]
            self.logging_rank = config["tracking"]["logging_rank"]
        else:
            self.config = None
            self.low_rank_cutoff = 0.1
            self.split_sigma = True
            self.logging_rank = 0
        self.trans = None
        self.shp = self.full_rank_weight.shape
        self.full_numel = self.full_rank_weight.numel()

    @torch.no_grad()
    def change_ab_train_mode(self, mode: str, cut_vals: bool):
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
            # if self.a.ndim > 1:
            weight = self.a @ self.b
            weight = (weight.T if self.trans else weight).reshape(self.shp).contiguous()
            self.full_rank_weight.set_(weight)

            if self.multi_dim_bias:
                bias = self.a_bias @ self.b_bias
                bias = (bias.T if self.bias_trans else bias).reshape(self.bias_shp).contiguous()

            self._set_req_grad_flags(True)
            return
        self.set_ab(mode=mode, cut_vals=cut_vals)
        self._set_req_grad_flags(True)

    @torch.no_grad()
    def set_ab(self, mode: str, cut_vals: bool = False):
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
            if dist.is_initialized():
                dist.all_reduce(cut, op=dist.ReduceOp.MAX)
            u = u[:, : cut + 1].clone()
            s = s[: cut + 1].clone()
            vh = vh[: cut + 1].clone()
            compression = ((s.shape[0] * (u.shape[0] + vh.shape[1])) / self.full_numel) * 100
            if dist.is_initialized() and dist.get_rank() == 0:
                log.info(
                    f"Change shape of orig: {tuple(self.shp)}, ({u.shape[0]}, {s.shape[0]}, {vh.shape[1]}), "
                    f"comp: {compression}",
                )

        if self.split_sigma:
            self.a.set_(u @ s.sqrt().diag())
            self.b.set_(s.sqrt().diag() @ vh)
        elif mode == "a":
            self.a.set_(u @ s.diag())
            self.b.set_(vh)
        else:  # if mode == "b": -> only other option....hopefully
            self.a.set_(u)
            self.b.set_(s.diag() @ vh)

        if self.multi_dim_bias:
            twodrepr, trans, _ = basis.get_2d_repr(self.full_rank_bias)
            if self.bias_trans is None:
                self.bias_trans = trans
            u, s, vh = torch.linalg.svd(twodrepr, full_matrices=False)
            if cut_vals:
                # TODO: logging
                cut = torch.nonzero(s > s[0] * self.low_rank_cutoff).flatten()[-1]
                if dist.is_initialized():
                    dist.all_reduce(cut, op=dist.ReduceOp.MAX)
                u = u[:, : cut + 1].clone()
                s = s[: cut + 1].clone()
                vh = vh[: cut + 1].clone()
                compression = ((s.shape[0] * (u.shape[0] + vh.shape[1])) / self.full_numel) * 100
                if dist.is_initialized() and dist.get_rank() == 0:
                    log.info(
                        f"Change shape of orig: {tuple(self.shp)}, ({u.shape[0]}, {s.shape[0]}, {vh.shape[1]}), "
                        f"comp: {compression}",
                    )
            if self.split_sigma:
                self.a_bias.set_(u @ s.sqrt().diag())
                self.b_bias.set_(s.sqrt().diag() @ vh)
            elif mode == "a":
                self.a_bias.set_(u @ s.diag())
                self.b_bias.set_(vh)
            else:  # if mode == "b": -> only other option....hopefully
                self.a_bias.set_(u)
                self.b_bias.set_(s.diag() @ vh)

    def _set_req_grad_flags(self, mode: bool = True):
        if self.training_mode_ab == "full":
            self.full_rank_weight.requires_grad = mode
            self.a.requires_grad = False
            self.b.requires_grad = False
        elif self.training_mode_ab == "a":
            self.full_rank_weight.requires_grad = False
            self.a.requires_grad = mode
            self.b.requires_grad = False
        elif self.training_mode_ab == "b":
            self.full_rank_weight.requires_grad = False
            self.a.requires_grad = False
            self.b.requires_grad = mode
        if self.multi_dim_bias:
            if self.training_mode_ab == "full":
                self.full_rank_bias.requires_grad = mode
                self.a_bias.requires_grad = False
                self.b_bias.requires_grad = False
            elif self.training_mode_ab == "a":
                self.full_rank_bias.requires_grad = False
                self.a_bias.requires_grad = mode
                self.b_bias.requires_grad = False
            elif self.training_mode_ab == "b":
                self.full_rank_bias.requires_grad = False
                self.a_bias.requires_grad = False
                self.b_bias.requires_grad = mode

    def train(self, mode: bool):
        super().train(mode)
        # call on internal layers, then set the weights and such to be what we want
        self._set_req_grad_flags(mode)

    def set_weight(self):
        if self.training_mode_ab == "full":
            self.base_layer.weight = self.full_rank_weight
        else:
            weight = self.a @ self.b
            weight = (weight.T if self.trans else weight).view(self.shp)
            self.full_rank_weight.set_(weight)
            self.base_layer.weight = self.full_rank_weight

        if self.multi_dim_bias:
            if self.training_mode_ab == "full":
                self.base_layer.bias = self.full_rank_bias
            else:
                bias = self.a @ self.b
                bias = (bias.T if self.bias_trans else bias).view(self.bias_shp)
                self.full_rank_bias.set_(bias)
                self.base_layer.bias = self.full_rank_bias

    def compare_bases(self, n: str = None):
        if not dist.is_initialized():
            return
        rank = dist.get_rank()
        ws = dist.get_world_size()

        # need to compare all ranks with all the others and then aggregate the results somehow...
        # also, if some of the basis lines up need to check to see where (since SVD is sorted)
        #   lets just check to see IF they actually line up at all for now
        cossim = nn.CosineSimilarity(dim=0)
        # get 2D USV rep -----------------------
        hld = self.a @ self.b
        u, _, vh = torch.linalg.svd(hld, full_matrices=False)
        uvh = u @ vh
        # --------------------------

        buff = torch.zeros((ws, *list(uvh.shape)), device=u.device, dtype=u.dtype)
        buff[rank] = uvh
        dist.all_reduce(buff)
        if rank == 0:
            log.info(f"Comparisons {n}")

        msgs = []
        sims = []
        for i, j in itertools.combinations(range(ws), 2):
            sim = cossim(buff[i], buff[j])
            sims.append(sim.mean())
            # if rank == 0:
            top5 = [f"{tf:.4f}" for tf in sim[:5]]
            msgs.append(f"ranks: {i} {j} - Similarity - {sim.mean():.4f}, top 5: {top5}")
        # if rank == 0 and sum(sims) / len(sims) < 0.9:
        #     for m in msgs:
        #         log.info(m)
        if rank == 0:
            log.info(f"Average sim across procs: {sum(sims) / len(sims)}")

    def get_compression(self):
        trainable = 0
        total = 0
        for n, p in self.named_parameters():
            if p.requires_grad:
                trainable += p.numel()
            if p.ndim == 1:
                total += p.numel()
        full_rank = self.full_rank_weight.numel()
        total += self.a.numel() + self.b.numel()
        if self.multi_dim_bias:
            full_rank += self.full_rank_bias.numel()
            total += self.a_bias.numel() + self.b_bias.numel()
        return trainable, total, full_rank


class ABLinear(ABLowRankLayer, nn.Linear):
    def __init__(self, existing_layer: nn.Linear, config: dict) -> None:
        if not isinstance(existing_layer, nn.Linear):
            raise TypeError("Only for Linear layers")
        nn.Linear.__init__(
            self,
            in_features=existing_layer.in_features,
            out_features=existing_layer.out_features,
            bias=existing_layer.bias is not None,
        )
        self.weight = existing_layer.weight
        self.bias = existing_layer.bias
        ABLowRankLayer.__init__(self, base_layer=existing_layer, config=config)
        del self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training_mode_ab == "full":
            weight = self.full_rank_weight
        else:
            weight = self.a @ self.b
            weight = (weight.T if self.trans else weight).view(self.shp)

        return nn.functional.linear(input, weight, self.bias)


class ABConv(ABLowRankLayer, nn.modules.conv._ConvNd):
    def __init__(self, existing_layer: nn.modules.conv._ConvNd, config: dict) -> None:
        if not isinstance(existing_layer, nn.modules.conv._ConvNd):
            raise TypeError("Only for Conv layers")
        nn.modules.conv._ConvNd.__init__(
            self,
            in_channels=existing_layer.in_channels,
            out_channels=existing_layer.out_channels,
            kernel_size=existing_layer.kernel_size,
            stride=existing_layer.stride,
            padding=existing_layer.padding,
            dilation=existing_layer.dilation,
            transposed=existing_layer.transposed,
            output_padding=existing_layer.output_padding,
            groups=existing_layer.groups,
            bias=existing_layer.bias is not None,
            padding_mode=existing_layer.padding_mode,
        )
        self.weight = existing_layer.weight
        self.bias = existing_layer.bias
        ABLowRankLayer.__init__(self, base_layer=existing_layer, config=config)
        self._conv_forward = existing_layer._conv_forward
        del self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training_mode_ab == "full":
            weight = self.full_rank_weight
        else:
            weight = self.a @ self.b
            weight = (weight.T if self.trans else weight).view(self.shp)
        return self._conv_forward(input, weight, self.bias)
