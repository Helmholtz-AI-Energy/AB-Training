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


class PatchworkSVDTrainer(BasicTrainer):
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
        self.warmup_steps = config.training.patchwork_svd.warmup_steps
        self.steps_btw_syncing = config.training.patchwork_svd.steps_btw_syncing

        if config.training.patchwork_svd.names_to_always_sync is None:
            names_to_always_sync = []
        else:
            names_to_always_sync = config.training.patchwork_svd.names_to_always_sync

        # names to always sync -> sync every step
        self.names_to_always_sync = names_to_always_sync
        self.all_names = []
        # names to not sync during individual training
        self.names_not_sync_in_indiv = []
        if names_to_always_sync is not None:
            for n, _ in model.named_parameters():
                self.all_names.append(n)
                if n not in names_to_always_sync or len(names_to_always_sync) == 0:
                    # have to build inverse of list
                    self.names_not_sync_in_indiv.append(n)

        self.comm_method = config.training.patchwork_svd.comm_method

        self.local_model = self.model

        # for 1d catting
        self.cat1d = config.training.patchwork_svd.cat1d
        if self.cat1d:
            self.names_to_cat1d, self.names_to_average = basis.get_1d_associated_weights(
                self.model,
                names_to_ignore=names_to_always_sync,
            )
            self.cat1d_dims = {}
        else:
            self.names_to_average = []
            for n, p in self.model.named_parameters():
                if p.squeeze().ndim < 2:
                    self.names_to_average.append(n)

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

                        hld = u @ new_s.diag() @ vh
                        if trans:
                            hld = hld.T
                        hld = hld.view(shp)
                        p.zero_()
                        p.add_(hld)

        if self.warmup_steps > 0 and len(self.names_to_always_sync) == 0:
            self.model_to_run = ddp_model  # ddp on all ranks

    @torch.no_grad()
    def _parchwork_sync(self):
        # this is the actual sync bit here:
        # options:
        #   partner - trade lowest USVh portions with highest from the partner (one partner for everything, diff partners each time)
        #   all-to-all - new weight matrix build from TOP USVh portions from all (equal contributions)
        #   one-to-all - replace the lowest N on all ranks with the top USVh portion of ONE rank

        pass

    @torch.no_grad()
    def _partner_sync_cat1d(self, partner: int, percent_to_send: float):
        # trade the percent to send with the partner from all ranks

        # trading is on dim0 for U and V (making it dim1 for Vh)

        # steps:
        #   1. get 2d reps (if 1d, go through that)
        #   2. get amount to send + send nonblocking, U will take longest, but with all the SVDs, it probably doesnt matter
        #   3. recv from partner, create new USVh
        #   4. set new USVh to model
        if percent_to_send > 1:
            raise ValueError(f"Percent to send must be < 1 -> {percent_to_send}")

        # TODO: check memory hit, might be very high if keeping all the USVh mats
        # my_usvh = {}
        # if self.cat1d:
        tag = 0
        for n in self.names_to_cat1d:
            base_weight = utils.rgetattr(self.model, n)
            base_2d_repr, trans, shp = basis.get_2d_repr(base_weight)
            catted, cat_names_n = basis.get_2d_rep_w_1d_names(
                model=self.model,
                base_2d=base_2d_repr,
                base_name=n,
                names1d=self.names_to_cat1d[n],
            )
            self.cat1d_dims[n] = cat_names_n
            u, s, vh = torch.linalg.svd(catted, full_matrices=False)
            # need buffer for u, s, and vh comms
            num_to_send = int(s.shape[0] * percent_to_send)
            send_u = u[:num_to_send].clone()
            send_s = s[:num_to_send].clone()
            send_vh = vh[:, :num_to_send].clone()

            recv_u, recv_s, recv_vh = torch.zeros_like(send_u), torch.zeros_like(send_s), torch.zeros_like(send_vh)

            send_u_op = dist.P2POp(dist.isend, send_u, partner, tag=tag)
            recv_u_op = dist.P2POp(dist.irecv, recv_u, partner, tag=tag + 1)
            send_s_op = dist.P2POp(dist.isend, send_s, partner, tag=tag + 2)
            recv_s_op = dist.P2POp(dist.irecv, recv_s, partner, tag=tag + 3)
            send_vh_op = dist.P2POp(dist.isend, send_vh, partner, tag=tag + 4)
            recv_vh_op = dist.P2POp(dist.irecv, recv_vh, partner, tag=tag + 5)
            tag += 6
            waits = dist.batch_isend_irecv([send_u_op, recv_u_op, send_s_op, recv_s_op, send_vh_op, recv_vh_op])

            # my_usvh[n] = {'u': u, 's': s, 'vh': vh, "trans": trans, "shp": shp, "waits": waits}
            # TODO: best way to distribute the workload here? 2 for loops to allow comes to finish?
            #    lets just make it blocking for now, uncomment later to test (also need to use self.my_usvh)
            # for n in self.names_to_cat1d:

            # u
            waits[0].wiat()
            waits[1].wait()
            u[-num_to_send:] = recv_u
            waits[2].wiat()
            waits[3].wait()
            s[-num_to_send:] = recv_s
            waits[4].wiat()
            waits[5].wait()
            vh[:, -num_to_send:] = recv_vh
            # undo 1d concatenating
            base_2d_repr, one_d_params = basis.get_1ds_from_2dcombi(
                u @ s.diag() @ vh,
                cat_dims=self.cat1d_dims[n],
            )
            # set 1D params
            for n1d in one_d_params:
                param = utils.rgetattr(self.model, n1d)
                param.zero_()
                param.add_(one_d_params[n1d])
            # set ND param
            if trans:
                base_2d_repr = base_2d_repr.T
            base_2d_repr = base_2d_repr.view(shp)
            base_weight.zero_()
            base_weight.add_(base_2d_repr)

    @torch.no_grad()
    def _partner_sync_no_cat1d(self, partner: int, percent_to_send: float):
        # trade the percent to send with the partner from all ranks

        # trading is on dim0 for U and V (making it dim1 for Vh)

        # steps:
        #   1. get 2d reps (if 1d, go through that)
        #   2. get amount to send + send nonblocking, U will take longest, but with all the SVDs, it probably doesnt matter
        #   3. recv from partner, create new USVh
        #   4. set new USVh to model
        if percent_to_send > 1:
            raise ValueError(f"Percent to send must be < 1 -> {percent_to_send}")

        # TODO: check memory hit, might be very high if keeping all the USVh mats
        # my_usvh = {}
        tag = 0
        for n, base_weight in self.model.named_parameters:
            if base_weight.ndim < 2:
                continue
            base_2d_repr, trans, shp = basis.get_2d_repr(base_weight)
            u, s, vh = torch.linalg.svd(base_2d_repr, full_matrices=False)
            # need buffer for u, s, and vh comms
            num_to_send = int(s.shape[0] * percent_to_send)
            send_u = u[:num_to_send].clone()
            send_s = s[:num_to_send].clone()
            send_vh = vh[:, :num_to_send].clone()

            recv_u, recv_s, recv_vh = torch.zeros_like(send_u), torch.zeros_like(send_s), torch.zeros_like(send_vh)

            send_u_op = dist.P2POp(dist.isend, send_u, partner, tag=tag)
            recv_u_op = dist.P2POp(dist.irecv, recv_u, partner, tag=tag + 1)
            send_s_op = dist.P2POp(dist.isend, send_s, partner, tag=tag + 2)
            recv_s_op = dist.P2POp(dist.irecv, recv_s, partner, tag=tag + 3)
            send_vh_op = dist.P2POp(dist.isend, send_vh, partner, tag=tag + 4)
            recv_vh_op = dist.P2POp(dist.irecv, recv_vh, partner, tag=tag + 5)
            tag += 6
            waits = dist.batch_isend_irecv([send_u_op, recv_u_op, send_s_op, recv_s_op, send_vh_op, recv_vh_op])

            # my_usvh[n] = {'u': u, 's': s, 'vh': vh, "trans": trans, "shp": shp, "waits": waits}
            # TODO: best way to distribute the workload here? 2 for loops to allow comes to finish?
            #    lets just make it blocking for now, uncomment later to test (also need to use self.my_usvh)
            # for n in self.names_to_cat1d:

            # u
            waits[0].wiat()
            waits[1].wait()
            u[-num_to_send:] = recv_u
            waits[2].wiat()
            waits[3].wait()
            s[-num_to_send:] = recv_s
            waits[4].wiat()
            waits[5].wait()
            vh[:, -num_to_send:] = recv_vh
            # undo 1d concatenating
            base_2d_repr = u @ s.diag() @ vh
            # set ND param
            if trans:
                base_2d_repr = base_2d_repr.T
            base_2d_repr = base_2d_repr.view(shp)
            base_weight.zero_()
            base_weight.add_(base_2d_repr)

    @torch.no_grad()
    def _partner_sync(self, partner: int, percent_to_send: float):
        # the functions will take care of the weights which they address, but need to find
        #   the weights which are not touched by them.
        waits = []
        for n in self.names_to_average:
            p = utils.rgetattr(self.model, n)
            p /= self.world_size
            waits.append(dist.all_reduce(p, async_op=True))

        if self.cat1d:
            self._partner_sync_cat1d(partner=partner, percent_to_send=percent_to_send)
        else:
            self._partner_sync_no_cat1d(partner=partner, percent_to_send=percent_to_send)

        for w in waits:
            w.wait()

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
                utils.reset_adam_state(self.optimizer, p)
            self.model_to_run = self.model
        elif (
            self.total_train_iterations % (self.steps_btw_syncing // 3) == 0
            and self.total_train_iterations > self.warmup_steps
        ):
            # finish average here -> receive everything - wait? then we dont need to do the weighted average...
            # for now, can just have this be blocking
            if self.rank == self.logging_rank:
                log.info(f"Compare sigma distribution: {self.current_iter}")

            # basis.compare_bases_across_ranks(self.model)

            for n, p in self.model.named_parameters():
                if n not in self.names_not_sync_in_indiv or p.ndim == 1:
                    continue
                else:
                    # print(f"sigma sync: {n}")
                    two_d_repr, trans, shp = basis.get_2d_repr(p)
                    u, s, vh = torch.linalg.svd(two_d_repr, full_matrices=False)

                    if self.rank == self.logging_rank:
                        nz10perc = torch.count_nonzero(s < s[0] * 0.1) / s.shape[0]
                        nz1perc = torch.count_nonzero(s < s[0] * 0.01) / s.shape[0]
                        nz50perc = torch.count_nonzero(s < s[0] * 0.5) / s.shape[0]

                        print(
                            f"{n}: {s.mean():.4f}, {s.min():.4f}, {s.max():.4f} "
                            f"-- <10% of first: {nz10perc:.4f} <1% of first: {nz1perc:.4f} "
                            f"<50% of first: {nz50perc:.4f}",
                        )
                utils.reset_adam_state(self.optimizer, p)
            self.model_to_run = self.model

    def _log_train(self, loss):
        # todo: metrics...
        if self.rank != self.logging_rank:
            return
        try:
            self.metrics.display(self.current_iter - 1)
        except AttributeError:
            pass
