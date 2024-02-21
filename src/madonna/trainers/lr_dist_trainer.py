import logging
import time
from copy import deepcopy

import scipy
import torch
import torch.distributed as dist
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from ..utils import basis, mixing, utils
from .base_trainer import BasicTrainer

log = logging.getLogger(__name__)


class LowRankSyncTrainer(BasicTrainer):
    def __init__(
        self,
        model: Module,
        optimizer: torch.optim.Optimizer,
        criterion: Module,
        device: torch.device,
        train_loader: DataLoader,
        config,
        lr_scheduler=None,
        metrics=None,
        # ==================================
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            use_autocast=config.model.autocast,
            max_grad_norm=config.training.max_grad_norm,
            iterations_per_train=config.training.iterations_per_train,
            log_freq=config.training.print_freq,
            logging_rank=config.tracking.logging_rank,
        )
        """
        1. warmup
            - init: ortho-sigma
            - full rank
            - normal ddp
        2. individual training w/o syncing
            - before X iterations: full rank
            - after X iterations: start transition to LR as in OIALR
                - should U/V be fixed globally or start locally
        3. measure similarity + average
            - in full rank: average is normal
            - in low-rank: only need to send sigma
            - should measure similarity here to show it
                - compare to:
                    - average weight matrix
                    - each other (requires more compute)
        4. repeat 2/3
        5. *at validation: make sure to average all models*
        """
        self.warmup_steps = config.training.syncing.warmup_steps
        # self.sync_freq = config.training.syncing.sync_freq
        self.steps_btw_syncing = config.training.syncing.steps_btw_syncing
        # which model to use: ????
        #   - need to set up multiple models
        #       - global DDP
        #       - local model
        #       - OIALR local
        #       - OIALR global
        if config.training.syncing.names_to_always_sync is None:
            names_to_always_sync = []
        else:
            names_to_always_sync = config.training.syncing.names_to_always_sync
        self.names_to_always_sync = names_to_always_sync
        self.all_names = []
        self.names_not_sync_in_indiv = []
        self.named_to_always_sync = []
        if names_to_always_sync is not None:
            self.named_to_always_sync = names_to_always_sync
            for n, _ in model.named_parameters():
                self.all_names.append(n)
                if n not in names_to_always_sync or len(names_to_always_sync) == 0:
                    # have to build inverse of list
                    self.names_not_sync_in_indiv.append(n)
        # if self.rank == 0:
        #     print(self.names_not_sync_in_indiv, names_to_always_sync)

        self.local_model = self.model
        # TODO: setup new init for rand-sigma
        self.config = config

        self.old_model = deepcopy(self.local_model)
        # start in warmup mode (standard global DDP)
        self.no_ddp = False
        if self.warmup_steps > 0 and len(self.names_to_always_sync) == 0:
            ddp_model = DDP(self.model)
            self.no_ddp = True

        if config.training.init_method == "rand-sigma":
            rank = dist.get_rank()
            local_seed = torch.random.initial_seed() + rank
            self.local_gen = torch.Generator(device=device).manual_seed(local_seed)
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in self.names_to_always_sync:
                        continue
                    two_d_repr, trans, shp = basis.get_2d_repr(p)
                    if two_d_repr is not None:
                        u, s, vh = torch.linalg.svd(two_d_repr, full_matrices=False)
                        order = torch.randperm(s.shape[0], generator=self.local_gen, device=s.device)
                        rand = torch.rand_like(s)
                        new_s = s[order] * rand
                        basis.get_og_repr(u @ new_s.diag() @ vh, trans, shp, set_loc=p)

        if self.warmup_steps > 0:
            self.model_to_run = ddp_model  # ddp on all ranks

    @torch.no_grad()
    def _pre_forward(self):
        # select model for forward step
        # if self.total_train_iterations < self.warmup_steps:
        #     self.model_to_run = self.models['global']
        # print(self.total_train_iterations, self.warmup_steps)
        if self.total_train_iterations == self.warmup_steps and not self.no_ddp:
            # if warmup is done, transition to individual mode
            self.model._ddp_params_and_buffers_to_ignore = self.names_not_sync_in_indiv
            # print(self.model._ddp_params_and_buffers_to_ignore)
            self.model_to_run = DDP(self.model) if len(self.names_to_always_sync) > 0 else self.model
            # print(self.model_to_run)
        elif (
            self.total_train_iterations % self.steps_btw_syncing == 0
            and self.total_train_iterations > self.warmup_steps
        ):
            # finish average here -> receive everything - wait? then we dont need to do the weighted average...
            # for now, can just have this be blocking
            if self.rank == self.logging_rank:
                log.info("Sycning ranks - details in config")

            basis.compare_bases_across_ranks(self.model)

            waits = {}
            # rank = dist.get_rank()
            # print(dist.get_rank(), self.logging_rank)
            # if dist.get_rank() == self.logging_rank:
            #     log.info(self.total_train_iterations)
            #     log.info("Sim to old model:")
            #     basis.compare_bases_baseline(self.model, self.old_model, print_rank=self.logging_rank)
            svd_stuff = {}
            for n, p in self.model.named_parameters():
                if n not in self.names_not_sync_in_indiv or p.ndim == 1:
                    # print(f"full sync: {n}")
                    p.data /= dist.get_world_size()  # use full world size
                    waits[n] = [dist.all_reduce(p.data, async_op=True), False, p.data]
                    continue
                else:
                    # print(f"sigma sync: {n}")
                    two_d_repr, trans, shp = basis.get_2d_repr(p)
                    u, s, vh = torch.linalg.svd(two_d_repr, full_matrices=False)
                    # s.data /= dist.get_world_size()  # use full world size
                    svd_stuff[n] = (u, s, vh, trans, shp)
                    # waits[n] = [dist.all_reduce(s.data, async_op=True), True, p.data]
                    waits[n] = [None, True, p.data]
            for n in waits:
                w, t, p = waits[n]
                if w is not None:
                    w.wait()
                if t:
                    u, s, vh, trans, shp = svd_stuff[n]

                    # cut out 1/world_size of the singular values
                    if self.world_size > 1:
                        cutoff = int(s.shape[0] * 0.75)  # / self.world_size)
                        u[-cutoff:] *= 0
                        s[-cutoff:] *= 0
                        vh[:, -cutoff:] *= 0
                        # if self.rank == self.logging_rank:
                        #     print(cutoff, s.shape[0])
                    hld = u @ s.diag() @ vh
                    if trans:
                        hld = hld.T
                    hld = hld.view(shp)
                    p.zero_()
                    p.add_(hld)

                    if self.rank == self.logging_rank:
                        nz10perc = torch.count_nonzero(s < s[0] * 0.1) / s.shape[0]
                        nz1perc = torch.count_nonzero(s < s[0] * 0.01) / s.shape[0]
                        nz50perc = torch.count_nonzero(s < s[0] * 0.5) / s.shape[0]

                        print(
                            f"{n}: {s.mean():.4f}, {s.min():.4f}, {s.max():.4f} "
                            f"-- <10% of first: {nz10perc:.4f} <1% of first: {nz1perc:.4f} "
                            f"<50% of first: {nz50perc:.4f}",
                        )

                    # basis.get_og_repr((u @ s.diag()) @ vh, trans, shp, set_loc=p)

                # w.wait()
                utils.reset_adam_state(self.optimizer, p)
            self.model_to_run = self.model

            # # local shuffle of sigma
            # with torch.no_grad():
            #     for n, p in self.model.named_parameters():
            #         if n in self.names_to_always_sync:
            #             continue
            #         two_d_repr, trans, shp = basis.get_2d_repr(p)
            #         if two_d_repr is not None:
            #             u, s, vh = torch.linalg.svd(two_d_repr, full_matrices=False)
            #             # mixing.exp_update_usvh_mix_sigma(u, s, vh, 1)
            #             # basis.get_og_repr(u @ s @ vh, trans, shp, set_loc=p)
            #             order = torch.randperm(s.shape[0], generator=self.local_gen, device=s.device)
            #             new_s = s[order]
            #             basis.get_og_repr(u @ new_s.diag() @ vh, trans, shp, set_loc=p)

            # if dist.get_rank() == self.logging_rank:
            #     log.info("Sim avg model v old:")
            #     basis.compare_bases_baseline(self.model, self.old_model, print_rank=self.logging_rank)
            # self.old_model = deepcopy(self.model)
        # pass

    def _log_train(self, loss):
        # todo: metrics...
        if self.rank != self.logging_rank:
            return
        try:
            self.metrics.display(self.current_iter - 1)
        except AttributeError:
            pass
