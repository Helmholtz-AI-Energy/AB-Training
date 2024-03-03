import logging
import time
from copy import deepcopy

import scipy
import torch
import torch.distributed as dist
import torch.nn.utils.prune as prune
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from ..models import ab_lr
from ..utils import basis, mixing, rgetattr, utils
from .base_trainer import BasicTrainer

log = logging.getLogger(__name__)


class ABLowRankTrainer(BasicTrainer):
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
        self.config = config
        """
        """
        self.warmup_steps = config.training.ab.warmup_steps
        self.steps_btw_syncing = config.training.ab.steps_btw_syncing
        self.ddp_steps_after_sync = config.training.ab.ddp_steps_after_sync
        self.zero_small_sig_vals = config.training.ab.zero_small_sig_vals
        self.use_pruning = config.training.ab.use_pruning
        self.reset_lr_on_sync = config.training.ab.reset_lr_on_sync

        # ---------------- MISC THINGS ---------------------------------------------------------
        self.comm_generator = torch.Generator().manual_seed(123456)  # no device -> use cpu
        self.local_model = self.model
        # --------------------------------------------------------------------------------------

        self.model = ab_lr.convert_network_ab_lowrank(self.model, config)

        # ------------------------ SYNC / COMM MODES -------------------------------------
        self.sync_mode = config.training.ab.sync_mode
        if self.sync_mode not in ["full", "with-stale", "only-learned"]:
            raise ValueError(f"sync mode must be one of [full, with-stale, only-learned], current: {self.sync_mode}")
        # for more info on how syncing works, see self._sync_processes

        # TODO: clean me up...
        self.ab_train_comm_setup = config.training.ab.train_comm_setup
        if self.ab_train_comm_setup not in ["individual", "groups"]:
            raise ValueError(f"AB train mode must be one of [individual, groups], current: {self.ab_train_comm_setup}")

        if config.training.ab.training_mode == "groups":
            self.use_groups = True
        else:
            self.use_groups = False

        # create the sub-groups in all cases, doesnt hurt (TODO: check that...)
        all_ranks = list(range(0, dist.get_world_size()))

        # Create two new groups of equal size
        self.group_a_ranks = all_ranks[0 : len(all_ranks) // 2]
        self.group_b_ranks = all_ranks[len(all_ranks) // 2 :]

        self.group_a = dist.new_group(ranks=self.group_a_ranks, use_local_synchronization=True)
        self.group_b = dist.new_group(ranks=self.group_b_ranks, use_local_synchronization=True)

        self.my_train_ab_mode = "a" if self.rank in self.group_a_ranks else "b"
        self.my_ab_group = self.group_a if self.rank in self.group_a_ranks else self.group_b

        if config.training.ab.full_rank_sync_names is None:
            full_rank_sync_names = []
        else:
            full_rank_sync_names = config.training.ab.full_rank_sync_names

        # names to always sync -> sync every step
        self.full_rank_sync_names = full_rank_sync_names
        self.all_names = []
        # names to not sync during individual training
        # self.ddp_params_and_buffers_to_ignore = []

        # Get lists of param names: always sync in full, 1d, multi-dim weights, a, b
        self.param_name_lists = {"always sync": [], "1d": [], "Nd": [], "a": [], "b": []}
        for n, p in model.named_parameters():
            if n in self.full_rank_sync_names or len(full_rank_sync_names) == 0:
                self.param_name_lists["always sync"].append(n)
            elif p.squeeze().ndim == 1:
                self.param_name_lists["1d"].append(n)
            elif n.endswith(".weight"):  # TODO: there should be a better rule here...
                # should be only weights with > 1D now, A/B will have different names
                self.param_name_lists["Nd"].append(n)
            elif n.endswith(".a"):
                self.param_name_lists["a"].append(n)
            elif n.endswith(".b"):
                self.param_name_lists["b"].append(n)
            else:
                raise ValueError(f"name: {n} not accounted for in lists")
        # --------------------------------------------------------------------------------

        # -------------------------------- DDP + WARMUP SETUP ------------------------------
        self.comm_reset_opt_on_sync = config.training.ab.reset_opt

        # start in warmup mode (standard global DDP)
        self.sync_bn = config.training.sync_batchnorm

        if self.sync_bn:
            self.sbn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(device)
            self.model_to_run = self.sbn_model

        if self.warmup_steps > 0:
            # starting in warmup phase
            templ = self.param_name_lists["a"] + self.param_name_lists["b"]
            self.sbn_model._ddp_params_and_buffers_to_ignore = templ
            ab_lr.change_ab_train_mode(  # TODO: test me, unsure if this will fail
                self.sbn_model,
                ab_training_mode="full",
                cut_singular_values=False,
            )
            ddp_model = DDP(self.sbn_model, process_group=None)
        else:
            # no warmup, starting in AB training
            # TODO: set up logic for different syncing modes
            ab_lr.change_ab_train_mode(
                self.sbn_model,
                ab_training_mode=self.my_train_ab_mode,
                cut_singular_values=False,
            )

            ign = self.param_name_lists["Nd"] + self.param_name_lists["a" if self.my_ab_group == "b" else "b"]
            self.sbn_model._ddp_params_and_buffers_to_ignore = ign
            ddp_model = DDP(self.sbn_model, process_group=self.my_ab_group)

        if self.warmup_steps > 0 or len(self.names_to_always_sync) != 0:
            self.model_to_run = ddp_model  # ddp on all ranks
        # ----------------------------------------------------------------------------------

        # ------------------------- LR Rebound ------------------------------
        self.in_lr_rebound = False
        self.lr_rebound_steps = config.training.ab.lr_rebound_steps
        self.current_lr_rebound_step = 0
        self.lr_rebound_step_factor = [0] * len(self.optimizer.param_groups)
        self.warmup_lr = config.training.lr_schedule.warmup_lr
        # -------------------------------------------------------------------

    def _prep_ab_train(self, cut_singular_values=False):
        # TODO: fix the DDP issue in the model/ddp model/local model.....im getting tired
        if isinstance(self.model_to_run, DDP):
            model = self.model_to_run.module
        else:
            model = self.model_to_run
        ab_lr.change_ab_train_mode(
            model,
            ab_training_mode=self.my_train_ab_mode,
            cut_singular_values=cut_singular_values,
        )
        ign = self.param_name_lists["Nd"]
        if self.ab_train_comm_setup == "groups":
            ign += self.param_name_lists["a" if self.my_ab_group == "b" else "b"]
        elif self.ab_train_comm_setup == "individual":
            ign += self.param_name_lists["a"] + self.param_name_lists["b"]
        else:
            raise ValueError(
                f"self.ab_train_comm_setup must be either 'groups' or 'individual': current {self.ab_train_comm_setup}",
            )

        model._ddp_params_and_buffers_to_ignore = ign
        return DDP(model, process_group=self.my_ab_group)

    @torch.no_grad()
    def _sync_processes(self):
        # if in groups: need to average all non-AB ranks as well
        waits = []
        if self.ab_train_comm_setup == "groups":
            names_to_avg_1d = self.param_name_lists["1d"] + self.param_name_lists["always sync"]
            for n, p in self.model_to_run.named_parameters():
                if n in names_to_avg_1d:
                    waits.append(dist.all_reduce(p, op=dist.ReduceOp.AVG, async_op=True))
        # three options: full, with-stale, and only-learned
        # full mode: -> transition network to 'full' mode, do average
        if self.sync_mode == "full":
            waits = self._full_sync(waits)
        # TODO: finish me!!
        # raise ValueError
        for w in waits:
            w.wait()

    def _full_sync(self, waits: list) -> list:
        # full mode: -> transition network to 'full' mode, do average
        ab_lr.change_ab_train_mode(
            self.model_to_run,
            ab_training_mode="full",
            cut_singular_values=False,
        )
        for n, p in self.model_to_run.named_parameters():
            if n in self.param_name_lists["Nd"]:
                waits.append(dist.all_reduce(p, op=dist.ReduceOp.AVG, async_op=True))

    def _setup_lr_rebound(self):
        self.in_lr_rebound = True
        self.current_lr_rebound_step = 0
        for c, pg in enumerate(self.optimizer.param_groups):
            target_lr = self.lr_scheduler._get_lr(self.lr_updates + self.lr_rebound_steps)[c]
            self.lr_rebound_step_factor[c] = (target_lr - self.warmup_lr) / self.lr_rebound_steps
            pg["lr"] = self.warmup_lr

    def _pre_forward(self):  # OVERWRITE base class
        # ---------------------- LR Rebound ----------------------------------------
        if not self.in_lr_rebound:
            return
        # if not in reboud, act normally. otherwise, increase the lr by lr_rebound_factor
        self.current_lr_rebound_step += 1
        for c, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self.lr_rebound_step_factor[c] * self.current_lr_rebound_step
            if self.logging_rank == self.rank and self.current_lr_rebound_step % 10 == 0:
                log.info(f"In LR rebound: actual LR: {pg['lr']:.6f}")
        if self.current_lr_rebound_step >= self.lr_rebound_steps:
            self.in_lr_rebound = False
        # --------------------------------------------------------------------------

    @torch.no_grad()
    def _post_train_step(self):
        # def _pre_forward(self):
        # select model for forward step
        # if self.total_train_iterations < self.warmup_steps:
        #     self.model_to_run = self.models['global']
        # print(self.total_train_iterations, self.warmup_steps)

        # TODO: remove no_ddp
        if self.total_train_iterations == self.warmup_steps and not self.no_ddp:
            if self.rank == self.logging_rank:
                log.info(
                    f"End of Warmup: {self.total_train_iterations} current epoch: {self.current_iter}/{self.iterations_per_train}",
                )
            # if warmup is done, transition to individual mode
            if isinstance(self.model_to_run, DDP):
                self.model_to_run = self.model_to_run.module
            self.model_to_run._ddp_params_and_buffers_to_ignore = self.ddp_params_and_buffers_to_ignore
            # if self.rank == self.logging_rank:
            #     print(self.ddp_params_and_buffers_to_ignore, type(self.model_to_run))
            self.model_to_run = DDP(self.model_to_run)

            if self.comm_reset_opt_on_sync:
                for p in self.model_to_run.parameters():
                    utils.reset_adam_state(self.optimizer, p)
            if self.reset_lr_on_sync:
                self._setup_lr_rebound()

        elif (
            self.total_train_iterations % self.steps_btw_syncing == 0
            and self.total_train_iterations > self.warmup_steps
        ):
            # # finish average here -> receive everything - wait? then we dont need to do the weighted average...
            # # for now, can just have this be blocking

            if self.rank == self.logging_rank:
                log.info(
                    f"Sycning ranks - iteration: {self.total_train_iterations} "
                    f"current epoch: {self.current_iter}/{self.iterations_per_train}",
                )

            # # # basis.compare_bases_across_ranks(self.model)
            # self._parchwork_sync()

            # -------- undo pruning and add models together -----------
            if self.use_pruning:
                reset_model_pruning(self.model_to_run)
            basis.compare_bases_across_ranks(self.model_to_run)

            for n, p in list(self.model_to_run.named_parameters()):
                # parts = n.split(".")
                # module_n = ".".join(parts[:-1])
                # if n not in self.ddp_params_and_buffers_to_ignore or p.ndim == 1:
                # dist.all_reduce(p, op=dist.ReduceOp.AVG)
                dist.all_reduce(p, op=dist.ReduceOp.AVG)
                if p.ndim > 1 and self.zero_small_sig_vals:
                    two_d_repr, trans, shp = basis.get_2d_repr(p)
                    u, s, vh = torch.linalg.svd(two_d_repr, full_matrices=False)
                    nz10 = torch.nonzero(s < s[0] * 0.1)
                    if self.rank == self.logging_rank:
                        # print(two_d_repr[:10, :10])
                        nz10perc = torch.count_nonzero(s < s[0] * 0.1) / s.shape[0]
                        nz1perc = torch.count_nonzero(s < s[0] * 0.01) / s.shape[0]
                        nz50perc = torch.count_nonzero(s < s[0] * 0.5) / s.shape[0]

                        # print(
                        #     f"{n}: {s.mean():.4f}, {s.min():.4f}, {s.max():.4f} "
                        #     f"-- <10% of first: {nz10perc:.4f} <1% of first: {nz1perc:.4f} "
                        #     f"<50% of first: {nz50perc:.4f}",
                        # )
                    if self.total_train_iterations >= self.steps_btw_syncing * 2:
                        if self.rank == self.logging_rank:
                            log.info(f"Removing {nz10perc * 100:.4f}% of sigma vals")
                        u[:, nz10] *= 0
                        s[nz10] *= 0
                        vh[nz10] *= 0
                        # vh[:, int(vh.shape[0] * self.comm_percent_to_send):] *= 0
                        hld = u @ s.diag() @ vh
                        if trans:
                            hld = hld.T
                        hld = hld.view(shp)
                        p.zero_()
                        p.add_(hld)
                if self.comm_reset_opt_on_sync:
                    utils.reset_adam_state(self.optimizer, p)

            if isinstance(self.model_to_run, DDP):
                self.model_to_run = self.model_to_run.module
            self.model_to_run._ddp_params_and_buffers_to_ignore = []
            self.model_to_run = DDP(self.model_to_run)

            if self.reset_lr_on_sync:
                self._setup_lr_rebound()
        elif (
            self.total_train_iterations > self.warmup_steps
            and self.total_train_iterations % self.steps_btw_syncing == self.ddp_steps_after_sync
            # self.total_train_iterations % (self.steps_btw_syncing // 2) == 0
            # and self.total_train_iterations > self.warmup_steps
        ):
            # finish average here -> receive everything - wait? then we dont need to do the weighted average...
            # for now, can just have this be blocking
            if self.rank == self.logging_rank:
                log.info(
                    f"SWITCH TO INDV TRAINING\nCompare sigma distribution: {self.total_train_iterations} "
                    f"current epoch: {self.current_iter}/{self.iterations_per_train}",
                )

            basis.compare_bases_across_ranks(self.model_to_run)

            # pruning tests: --------------------------
            # if self.rank == self.logging_rank:
            #     log.info(f"Removing sigma vals")

            if isinstance(self.model_to_run, DDP):
                self.model_to_run = self.model_to_run.module
            self.model_to_run._ddp_params_and_buffers_to_ignore = self.ddp_params_and_buffers_to_ignore
            self.model_to_run = DDP(self.model_to_run)

            for n, p in list(self.model_to_run.named_parameters()):
                # parts = n.split(".")
                # module_n = ".".join(parts[:-1])
                if p.ndim > 1:
                    two_d_repr, trans, shp = basis.get_2d_repr(p)
                    u, s, vh = torch.linalg.svd(two_d_repr, full_matrices=False)
                    # nz10 = torch.nonzero(s < s[0] * 0.1)
                    if self.rank == self.logging_rank:
                        # print(two_d_repr[:10, :10])
                        nz10perc = torch.count_nonzero(s < s[0] * 0.1) / s.shape[0]
                        nz1perc = torch.count_nonzero(s < s[0] * 0.01) / s.shape[0]
                        nz50perc = torch.count_nonzero(s < s[0] * 0.5) / s.shape[0]

                        print(
                            f"{n}: {s.mean():.4f}, {s.min():.4f}, {s.max():.4f} "
                            f"-- <10% of first: {nz10perc:.4f} <1% of first: {nz1perc:.4f} "
                            f"<50% of first: {nz50perc:.4f}",
                        )
                    # if self.total_train_iterations > self.steps_btw_syncing * 2:
                    #     if self.rank == self.logging_rank:
                    #         log.info(f"Removing {nz10perc * 100:.4f}% of sigma vals")
                    #     u[:, nz10] *= 0
                    #     s[nz10] *= 0
                    #     vh[nz10] *= 0
                    #     # vh[:, int(vh.shape[0] * self.comm_percent_to_send):] *= 0
                    #     hld = u @ s.diag() @ vh
                    #     if trans:
                    #         hld = hld.T
                    #     hld = hld.view(shp)
                    #     p.zero_()
                    #     p.add_(hld)

                if self.comm_reset_opt_on_sync:
                    utils.reset_adam_state(self.optimizer, p)

            if self.reset_lr_on_sync:
                self._setup_lr_rebound()

    def _log_train(self, loss):
        # todo: metrics...
        if self.rank != self.logging_rank:
            return
        try:
            self.metrics.display(self.current_iter - 1)
        except AttributeError:
            pass
        out_lrs = []
        prnt_str = "LRs: "
        for group in self.optimizer.param_groups:
            out_lrs.append(group["lr"])
            prnt_str += f"group {len(out_lrs)}: lr {group['lr']:.6f}\t"
        print(prnt_str)
        # return out_lrs, prnt_str


def get_prune_mask_from_2d_repr(twodimrepr, trans, shp, start, stop, dim=0):
    # TODO: check that '1' means 'keep' in prune
    mask = torch.zeros_like(twodimrepr, dtype=torch.int)
    sl = [slice(None)] * twodimrepr.ndim
    sl[dim] = slice(start, stop)
    mask[sl] = 1  # set section of 2d repr to 0 to prune it off
    # percent = 0.25
    # mask = (torch.rand_like(twodimrepr) > percent).to(torch.bool)
    if trans and twodimrepr.ndim > 1:
        mask = mask.T
    return mask.reshape(shp)


def reset_model_pruning(model):
    for module in model.modules():
        try:
            prune.remove(module, "weight")
        except ValueError:
            # no pruning on this layer
            pass
