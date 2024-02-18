import logging
import time
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from ..utils import basis, utils
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
        # ==================================
        warmup_steps: int,
        # ==================================
        lr_scheduler=None,
        use_autocast: bool = False,
        max_grad_norm: float = 0,
        metrics: MetricCollection = None,
        iterations_per_train: int = None,
        log_freq: int = 20,
        # ==================================
        sync_freq: int = 5,
        steps_btw_syncing: int = 1000,
        names_to_always_sync: list = None,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            device,
            train_loader,
            lr_scheduler,
            use_autocast,
            max_grad_norm,
            metrics,
            iterations_per_train,
            log_freq,
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
        self.warmup_steps = warmup_steps
        self.sync_freq = sync_freq
        self.steps_btw_syncing = steps_btw_syncing
        # which model to use:
        #   - need to set up multiple models
        #       - global DDP
        #       - local model
        #       - OIALR local
        #       - OIALR global
        if names_to_always_sync is not None:
            self.all_names = []
            self.names_not_sync_in_indiv = []
            self.named_to_always_sync = names_to_always_sync
            for n, _ in model.named_parameters():
                self.all_names.append(n)
                if n not in names_to_always_sync:
                    # have to build inverse of list
                    self.names_not_sync_in_indiv.append(n)

        self.local_model = self.model
        self.old_model = deepcopy(self.local_model)
        # start in warmup mode (standard global DDP)
        if self.warmup_steps > 0:
            self.model_to_run = DDP(self.model)  # ddp on all ranks

    @torch.no_grad()
    def _pre_forward(self):
        # select model for forward step
        # if self.total_train_iterations < self.warmup_steps:
        #     self.model_to_run = self.models['global']
        if self.total_train_iterations == self.warmup_steps:
            # if warmup is done, transition to individual mode
            self.model._ddp_params_and_buffers_to_ignore = self.names_not_sync_in_indiv
            self.model_to_run = DDP(self.model)

        elif self.total_train_iterations % self.steps_btw_syncing == 0:
            # finish average here -> receive everything - wait? then we dont need to do the weighted average...
            # for now, can just have this be blocking
            waits = []
            # rank = dist.get_rank()

            log.info("Sim to old model:")
            basis.compare_bases_baseline(self.model, self.old_model)

            for n, p in self.model.named_parameters():
                if n not in self.names_not_sync_in_indiv:
                    p.data /= dist.get_world_size()  # use full world size
                    waits.append(dist.all_reduce(p.data, async_op=True))
            log.info("Sim avg model v old:")
            basis.compare_bases_baseline(self.model, self.old_model)
        # pass

    def _log_train(self, loss):
        # todo: metrics...
        log.info(f"{self.current_iter}/{self.iterations_per_train}: loss: {loss:.5f}")
        # return super()._log_train(loss)
