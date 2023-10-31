import logging
import time
from collections import OrderedDict, defaultdict
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import (
    change_adam_shapes,
    change_sgd_shapes,
    create_svd_param_groups,
    replace_opt_state_with_svd_adam,
)
from .attention import SVDSyncMultiheadAttention
from .conv import SVDSyncConv1d, SVDSyncConv2d
from .linear import SVDSyncLinear

log = logging.getLogger(__name__)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class SVDSyncModel(nn.Module):
    def __init__(
        self,
        full_rank_model: nn.Module,
        keep_first_layer: bool = False,
        keep_last_layer: bool = True,
        step_on_forward: bool = True,
        full_rank_warmup: bool = False,  # TODO: remove me or deal with me somehow... not planed to use atm
        fixed_inner_dim: bool = True,
        inner_dim_init_ratio: float = 1.0,
        random_simga: bool = False,
        # --------- blur params ----------------------
        mixing_method: str = "exp",
        mixing_options: dict = None,
        # --------- sync params ----------------------
        sync_frequency: int = 1000,
        sync_delay: int = 1000,
        trade_method: str = "fib",
        vecs_to_trade: int = 100,
        ordering: str = "cat",
    ):
        """
        FIXME: do this stuffs

        This class is for the specific methods to sync SVD models
        """
        super().__init__()

        # ---------- model stuff --------------------
        num = 0
        full_params = 0
        for n, p in full_rank_model.named_parameters():
            if n[-2:] not in [".s", "_s", "_u", ".u", "vh"]:
                full_params += p.numel()
                if p.requires_grad:
                    # print(f"{n}: {p.numel()}")
                    num += p.numel()

        self.base_trainable_parameters = num
        self.base_all_parameters = full_params
        self.fixed_inner_dim = fixed_inner_dim
        self.inner_dim_init_ratio = inner_dim_init_ratio

        self.update_from_simga = True  # update_from_simga TODO: remove me?
        self.first_layer = keep_first_layer
        self.keep_last_layer = keep_last_layer
        self.last_layer = None
        self.full_rank_warmup = full_rank_warmup
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()
        self.local_full_rank_model = full_rank_model
        self.low_rank_replacement_list = {}
        self.local_low_rank_model = None
        # self.non_svd_params = []
        self.random_simga = random_simga

        if full_rank_warmup and sync_delay > 0:
            if self.rank == 0:
                log.info("Starting with training in full rank")
            self.model = self.local_full_rank_model
        else:
            self.setup_low_rank_training(skip_optimizer_init=True)

        self.optimizer = None  # optimizers.MixedSVDOpt
        self.state_dict = self.model.state_dict
        self.parameters = self.model.parameters
        self.named_parameters = self.model.named_parameters
        self.named_modules = self.model.named_modules
        self.named_buffers = self.model.named_buffers
        self.named_children = self.model.named_children
        self.children = self.model.children
        self.cuda = self.model.cuda
        # TODO: add other methods here??
        self.__repr__ = self.model.__repr__
        self.train = self.model.train
        # ---------------- sync params -----------------------------------------
        self.sync_frequency = sync_frequency
        self.call_count = 0
        self.num_stability_layvers_to_check = 0
        self.delay = sync_delay
        # self.fib1, self.fib2 = 0, 1
        self.next_sync_iteration = self.delay + self.fib1
        self.trade_method = trade_method  # method for trading the singular values and vectors
        self.vecs_to_trade = vecs_to_trade  # number of vectors to send each time (upper limit)
        self.ordering = ordering  # how to order the vals/vecs
        # ---------------- mixing params  ------------------------------------
        self.mixing_method = mixing_method
        if mixing_options is None:
            mixing_options = {}
        self.mixing_options = mixing_options
        # ------------------- other params ------------------------------------
        self.local_generator = torch.Generator()
        self.local_generator.manual_seed(self.rank)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer  # optimizers.MixedSVDOpt
        if isinstance(optimizer, (optim.Adam, optim.AdamW)):
            self.reshape_opt_state_fn = change_adam_shapes
        elif isinstance(optimizer, optim.SGD):
            self.reshape_opt_state_fn = change_sgd_shapes

    @torch.no_grad()
    def setup_low_rank_training(self, skip_optimizer_init=False):
        """
        Sets up the low rank training.

        If the optimizer state is populated (full-rank training done before this is called),
        then skip_optimizer_init should be True. This will call another helper to change the
        optimizer states to align with the OIALR weight shapes.

        Args:
                skip_optimizer_init: If True don't initialize SVD
        """
        if self.rank == 0:
            log.info("Starting with training in low rank")
        self.local_low_rank_model = self._replace_layers(self.local_full_rank_model)

        # Reset the last layer to the last layer.
        if self.keep_last_layer:
            self._reset_last_layer(self.local_low_rank_model)

        self.model = self.local_low_rank_model
        self.svd_modules = {}
        self.layer_names = []
        calls = 0
        sz = 1 if not self.use_ddp else dist.get_world_size()
        for name, mod in self.model.named_modules():
            if hasattr(mod, "test_stability_distributed"):
                if self.distributed_updates:
                    working_rank = calls % sz
                else:
                    working_rank = self.rank
                # self.svd_modules.append((name, mod, working_rank))
                self.svd_modules[name] = {"mod": mod, "working_rank": working_rank, "stable": False, "stable_delay": 0}
                self.layer_names.append(name)
                try:  # only a module in the attention layers and its just faster to try to get something
                    if mod._qkv_same_embed_dim:
                        calls += 1
                    else:
                        calls += 3
                except AttributeError:
                    calls += 1
        if not skip_optimizer_init:  # only need to do this if we start in full rank
            replace_opt_state_with_svd_adam(self.optimizer, self.low_rank_replacement_list)

        # self.non_svd_params = []
        # for n, p in self.named_parameters():
        #     if not n.endswith((".s", ".u", ".vh")) and p.requires_grad:
        #         self.non_svd_params.append(p)

    def mix_svd_layers(self):
        log.info(f"Mixing sigma of SVD layers with {self.mixing_method}")
        for name in self.svd_modules:
            self.svd_modules[name]["mod"].mix_simga(method=self.mixing_method, **self.mixing_options)

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        if isinstance(module, nn.Linear) and min(module.weight.shape) > max(module.weight.shape) / 10:
            if not self.first_layer:
                module_output = SVDSyncLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    uvhthreshold=self.uvhthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    start_weight=module.weight,
                    start_bias=module.bias,
                    update_from_simga=self.update_from_simga,
                    reinit_shapes=self.reinit_shapes,
                    distributed_updates=self.use_ddp,
                    inner_dim_init_ratio=self.inner_dim_init_ratio,
                    random_sigma=self.random_simga,
                ).to(device=module.weight.device, dtype=module.weight.dtype)
                self.last_layer = [module, name, module.weight.dtype, module.weight.device]
                self.low_rank_replacement_list[id(module.weight)] = [module_output.s, "lin"]
            else:
                self.first_layer = False
        elif isinstance(module, nn.MultiheadAttention):
            if not self.first_layer:
                module_output = SVDSyncMultiheadAttention(
                    embed_dim=module.embed_dim,
                    num_heads=module.num_heads,
                    dropout=module.dropout,
                    bias=module.in_proj_bias is not None,
                    add_bias_kv=module.bias_k is not None,
                    add_zero_attn=module.add_zero_attn,
                    kdim=module.kdim,
                    vdim=module.vdim,
                    batch_first=module.batch_first,
                    uvh_threshold=self.uvhthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    start_q=module.q_proj_weight,
                    start_k=module.k_proj_weight,
                    start_v=module.v_proj_weight,
                    start_in_proj=module.in_proj_weight,
                    start_k_bias=module.bias_k,
                    start_v_bias=module.bias_v,
                    start_in_proj_bias=module.in_proj_bias,
                    update_from_simga=self.update_from_simga,
                    reinit_shapes=self.reinit_shapes,
                    distributed_updates=self.use_ddp,
                    inner_dim_init_ratio=self.inner_dim_init_ratio,
                    random_sigma=self.random_simga,
                ).to(device=module.out_proj.weight.device, dtype=module.out_proj.weight.dtype)
                self.last_layer = [module, name, None, None]
                if module.in_proj_weight is not None:
                    self.low_rank_replacement_list[id(module.in_proj_weight)] = [module_output.in_proj_s, "attn"]
                else:
                    self.low_rank_replacement_list[id(module.q_proj_weight)] = [module_output.q_s, "attn"]
                    self.low_rank_replacement_list[id(module.k_proj_weight)] = [module_output.k_s, "attn"]
                    self.low_rank_replacement_list[id(module.v_proj_weight)] = [module_output.v_s, "attn"]
            else:
                self.first_layer = False
        elif isinstance(module, nn.Conv2d):
            wv = module.weight.view(module.weight.shape[0], -1)
            if wv.shape[0] < wv.shape[1]:
                wv.T
            if wv.shape[1] < wv.shape[0] / 10:
                pass  # skip this layer if there are not enough params
            elif not self.first_layer:
                module_output = SVDSyncConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    uvhthreshold=self.uvhthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    start_bias=module.bias,
                    start_weight=module.weight,
                    update_from_simga=self.update_from_simga,
                    reinit_shapes=self.reinit_shapes,
                    norm=module.norm if hasattr(module, "norm") else None,
                    activation=module.activation if hasattr(module, "activation") else None,
                    distributed_updates=self.use_ddp,
                    inner_dim_init_ratio=self.inner_dim_init_ratio,
                    random_sigma=self.random_simga,
                )
                self.last_layer = [module, name, module.weight.dtype, module.weight.device]
                self.low_rank_replacement_list[id(module.weight)] = [module_output.s, "conv"]
            else:
                self.first_layer = False
        elif isinstance(module, nn.Conv1d):
            wv = module.weight.view(module.weight.shape[0], -1)
            if wv.shape[0] < wv.shape[1]:
                wv.T
            if wv.shape[1] < wv.shape[0] / 10:
                pass  # skip this layer if there are not enough params
            elif not self.first_layer:
                module_output = SVDSyncConv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    uvhthreshold=self.uvhthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    start_bias=module.bias,
                    start_weight=module.weight,
                    update_from_simga=self.update_from_simga,
                    reinit_shapes=self.reinit_shapes,
                    norm=module.norm if hasattr(module, "norm") else None,
                    activation=module.activation if hasattr(module, "activation") else None,
                    distributed_updates=self.use_ddp,
                    inner_dim_init_ratio=self.inner_dim_init_ratio,
                    random_sigma=self.random_simga,
                )
                self.last_layer = [module, name, module.weight.dtype, module.weight.device]
                self.low_rank_replacement_list[id(module.weight)] = [module_output.s, "conv"]
            else:
                self.first_layer = False
        for n, child in module.named_children():
            module_output.add_module(
                f"{n}",
                self._replace_layers(
                    child,
                    name=f"{name}.{n}" if name is not None else f"{n}",
                    process_group=process_group,
                ),
            )
        del module
        return module_output

    def _reset_last_layer(self, module, name=None):
        module_output = module
        if name == self.last_layer[1]:
            if self.last_layer[2] is None:
                try:
                    device = module.in_proj_s.device
                    dtype = module.in_proj_s.dtype
                except AttributeError:
                    device = module.q_s.device
                    dtype = module.q_s.dtype
            else:
                dtype = self.last_layer[2]
                device = self.last_layer[3]
            module_output = self.last_layer[0].to(device=device, dtype=dtype)
            del self.low_rank_replacement_list[id(self.last_layer[0].weight)]
        for n, child in module.named_children():
            module_output.add_module(n, self._reset_last_layer(child, f"{name}.{n}" if name is not None else f"{n}"))
        # del module
        return module_output

    @torch.no_grad()
    def get_perc_params_all_layers(self):
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        # percs, actives, normals = [], [], []
        trainable = 0
        untrainable = 0
        for n, p in self.model.named_parameters():
            # print(f"{n} {p.requires_grad}")
            if p.requires_grad:
                # if n[-2:] not in [".s", "_s", "_u", ".u", "vh"]:
                trainable += p.numel()
            else:
                untrainable += p.numel()

        full_normal = self.base_trainable_parameters
        full_active = trainable
        # full_deactivated = untrainable
        full_model = trainable + untrainable
        compression_perc = 100 * (full_model / self.base_all_parameters)
        if rank == 0:
            log.info(
                f"Active Params: {100 * (full_active / full_normal):.4f}% active: {full_active} "
                f"Full Rank: {full_normal} Low rank total: {full_model} compression: {compression_perc}",
            )
        return 100 * (full_active / full_normal), full_active, full_normal, compression_perc

    @staticmethod
    def reset_all_opt_states(optimizer: optim.Optimizer):
        # reset op1 first
        # for group in self.opt1.param_groups:
        optimizer.state = defaultdict(dict)

    @torch.no_grad()  # The function is the main method of doing stability tracking
    def sync_nonsvd_params(self, nonblocking=True, waits=None):
        # TODO: make a buffer to hold all of the weights for this to make this faster
        #   (unsure if it will help)
        if not dist.is_initialized():
            return
        # for nonblocking case, check if there are waits to wait for
        # if nonblocking, wait for the sent items then exit
        if waits is not None:
            for w in waits:
                w.wait()
            if nonblocking:
                return None
        waits = []
        for n, p in self.named_parameters():
            if not p.requires_grad or n.endswith(("_u", ".u", "_vh", ".vh", ".s", "_s")):
                continue
            waits.append(dist.all_reduce(p, op=dist.ReduceOp.AVG, async_op=True))
        # if nonblocking, return the waits for later
        # if blocking, wait for the op to complete right now
        if nonblocking:
            return waits
        else:
            for w in waits:
                w.wait()

    def forward(self, *args, **kwargs):
        # TODO: if we want to run this every N steps, then we need to track all of that.
        #       also, need to make specific functions for the val iterations
        # if self.step_on_forward and self.model.training:
        #     # print("in stability tracking block")
        #     self.model_stability_tracking(force=False)
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def load_state_dict(self, best_model_path):
        # print(state_dict.keys())
        # return ValueError
        import os

        lcl_rank = int(os.environ["PMIX_RANK"])
        state_dict = torch.load(best_model_path, map_location=f"cuda:{lcl_rank}")
        for n, p in self.local_low_rank_model.named_parameters():
            # if self.local_low_rank_model[k]
            loaded_param = state_dict[n]
            # loaded_param = loaded_param.to(dtype=p.dtype, device=p.device)
            # print(k, '\t', n)

            if loaded_param.shape != p.shape:
                # print(f"changing shape of {n}")
                p.set_(torch.zeros(loaded_param.shape, dtype=p.dtype, device=p.device))

        self.local_low_rank_model.load_state_dict(state_dict)
        self.get_perc_params_all_layers()
