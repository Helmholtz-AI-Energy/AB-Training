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

from ..utils import basis, mixing, rgetattr, utils
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
        self.ddp_steps_after_sync = config.training.patchwork_svd.ddp_steps_after_sync
        self.zero_small_sig_vals = config.training.patchwork_svd.zero_small_sig_vals
        self.use_pruning = config.training.patchwork_svd.use_pruning
        self.reset_lr_on_sync = config.training.patchwork_svd.reset_lr_on_sync

        if config.training.patchwork_svd.names_to_always_sync is None:
            names_to_always_sync = []
        else:
            names_to_always_sync = config.training.patchwork_svd.names_to_always_sync

        # names to always sync -> sync every step
        self.names_to_always_sync = names_to_always_sync
        self.all_names = []
        # names to not sync during individual training
        self.ddp_params_and_buffers_to_ignore = []
        if names_to_always_sync is not None:
            for n, _ in model.named_parameters():
                self.all_names.append(n)
                if n not in names_to_always_sync or len(names_to_always_sync) == 0:
                    # have to build inverse of list
                    self.ddp_params_and_buffers_to_ignore.append(n)

        self.comm_method = config.training.patchwork_svd.comm_method
        if self.comm_method in ["partner", "one-to-all"]:
            self.comm_percent_to_send = config.training.patchwork_svd.comm_kwargs.percent_to_send
            self.comm_generator = torch.Generator().manual_seed(123456)  # no device -> use cpu
        else:
            self.comm_percent_to_send = 1 / self.world_size

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
            self.names_to_not_average = []
            for n, p in self.model.named_parameters():
                if p.squeeze().ndim < 2:
                    self.names_to_average.append(n)
                # else:
                #     self.names_to_not_average.append(n)

        self.comm_reset_opt_on_sync = config.training.patchwork_svd.reset_opt

        # start in warmup mode (standard global DDP)
        self.sync_bn = config.training.sync_batchnorm
        self.no_ddp = True

        if self.sync_bn:
            self.sbn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(device)

        if self.warmup_steps > 0:
            ddp_model = DDP(self.sbn_model)
            self.no_ddp = False
        elif len(self.names_to_always_sync) != 0:
            self.sbn_model._ddp_params_and_buffers_to_ignore = self.ddp_params_and_buffers_to_ignore
            ddp_model = DDP(self.sbn_model)
            self.no_ddp = False

        if config.training.init_method == "rand-sigma":
            rank = dist.get_rank()
            local_seed = torch.random.initial_seed() + rank
            self.local_gen = torch.Generator(device=device).manual_seed(local_seed)
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n not in self.ddp_params_and_buffers_to_ignore:  # in self.names_to_always_sync:
                        continue
                    two_d_repr, trans, shp = basis.get_2d_repr(p)
                    if two_d_repr is None:
                        continue
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
        elif config.training.init_method == "sloped-sigma":
            rank = dist.get_rank()
            local_seed = torch.random.initial_seed() + rank
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n not in self.ddp_params_and_buffers_to_ignore:  # in self.names_to_always_sync:
                        continue
                    two_d_repr, trans, shp = basis.get_2d_repr(p)
                    if two_d_repr is None:
                        continue
                    u, s, vh = torch.linalg.svd(two_d_repr, full_matrices=False)
                    decay_factor = torch.tensor(0.1)
                    exponents = torch.arange(s.shape[0], dtype=s.dtype, device=s.device)
                    new_s = s[0] * 2 * decay_factor**exponents

                    hld = u @ new_s.diag() @ vh
                    if trans:
                        hld = hld.T
                    hld = hld.view(shp)
                    p.zero_()
                    p.add_(hld)

        if self.warmup_steps > 0 or len(self.names_to_always_sync) != 0:
            self.model_to_run = ddp_model  # ddp on all ranks
        # if self.logging_rank == self.rank:
        #     print(self.model_to_run)

        # LR Rebound phase ------------------------------
        self.in_lr_rebound = False
        self.lr_rebound_steps = config.training.patchwork_svd.lr_rebound_steps
        self.current_lr_rebound_step = 0
        self.lr_rebound_step_factor = [0] * len(self.optimizer.param_groups)
        self.warmup_lr = config.training.lr_schedule.warmup_lr
        # -----------------------------------------------

    def _setup_lr_rebound(self):
        self.in_lr_rebound = True
        self.current_lr_rebound_step = 0
        for c, pg in enumerate(self.optimizer.param_groups):
            target_lr = self.lr_scheduler._get_lr(self.lr_updates + self.lr_rebound_steps)[c]
            self.lr_rebound_step_factor[c] = (target_lr - self.warmup_lr) / self.lr_rebound_steps
            pg["lr"] = self.warmup_lr

    @torch.no_grad()
    def _parchwork_sync(self):
        # this is the actual sync bit here:
        # options:
        #   partner - trade lowest USVh portions with highest from the partner (one partner for everything, diff partners each time)
        #   all-to-all - new weight matrix build from TOP USVh portions from all (equal contributions)
        #   one-to-all - replace the lowest N on all ranks with the top USVh portion of ONE rank
        tsync = time.perf_counter()
        if self.comm_method == "partner":
            # need generator
            # need partners
            partners = torch.randperm(self.world_size, generator=self.comm_generator).view(self.world_size // 2, 2)
            mypair = partners[torch.any(self.rank == partners, dim=1)]
            mypartner = mypair[mypair != self.rank].item()
            self._partner_sync(partner=mypartner, percent_to_send=self.comm_percent_to_send)

            log.info(f"Partner sync: partner -> {mypartner}, time: {time.perf_counter() - tsync}")
        elif self.comm_method == "all-to-all":
            self._all_to_all_sync()
            log.info(f"All equal shares sync. time: {time.perf_counter() - tsync}")
        elif self.comm_method == "one-to-all":
            root = torch.randint(self.world_size, (1,), generator=self.comm_generator).item()
            self._one_to_all_sync(root=root, percent_to_send=self.comm_percent_to_send)
            log.info(f"One to all sync: root -> {root}, time: {time.perf_counter() - tsync}")

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
                names1d=self.names_to_cat1d,
            )
            self.cat1d_dims[n] = cat_names_n
            u, s, vh = torch.linalg.svd(catted, full_matrices=False)
            # need buffer for u, s, and vh comms
            send_u = u[: int(u.shape[0] * percent_to_send)].contiguous()
            send_s = s[: int(s.shape[0] * percent_to_send)].contiguous()
            send_vh = vh[:, : int(vh.shape[0] * percent_to_send)].contiguous()

            recv_u, recv_s, recv_vh = torch.zeros_like(send_u), torch.zeros_like(send_s), torch.zeros_like(send_vh)

            send_u_op = dist.P2POp(dist.isend, send_u, partner, tag=tag)
            recv_u_op = dist.P2POp(dist.irecv, recv_u, partner, tag=tag + 1)
            send_s_op = dist.P2POp(dist.isend, send_s, partner, tag=tag + 2)
            recv_s_op = dist.P2POp(dist.irecv, recv_s, partner, tag=tag + 3)
            send_vh_op = dist.P2POp(dist.isend, send_vh, partner, tag=tag + 4)
            recv_vh_op = dist.P2POp(dist.irecv, recv_vh, partner, tag=tag + 5)
            tag += 6
            waits = dist.batch_isend_irecv([send_u_op, recv_u_op, send_s_op, recv_s_op, send_vh_op, recv_vh_op])
            waits[0].wait()

            # my_usvh[n] = {'u': u, 's': s, 'vh': vh, "trans": trans, "shp": shp, "waits": waits}
            # TODO: best way to distribute the workload here? 2 for loops to allow comes to finish?
            #    lets just make it blocking for now, uncomment later to test (also need to use self.my_usvh)
            # for n in self.names_to_cat1d:

            # u
            # waits[0].wait()
            # waits[1].wait()
            u[-int(u.shape[0] * percent_to_send) :] = recv_u
            # waits[2].wait()
            # waits[3].wait()
            s[-int(s.shape[0] * percent_to_send) :] = recv_s
            # waits[4].wait()
            # waits[5].wait()
            vh[:, -int(vh.shape[0] * percent_to_send) :] = recv_vh
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
                if self.comm_reset_opt_on_sync:
                    utils.reset_adam_state(self.optimizer, param)
            # set ND param
            if trans:
                base_2d_repr = base_2d_repr.T
            base_2d_repr = base_2d_repr.view(shp)
            base_weight.zero_()
            base_weight.add_(base_2d_repr)
            if self.comm_reset_opt_on_sync:
                utils.reset_adam_state(self.optimizer, base_2d_repr)

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
        for n, base_weight in self.model.named_parameters():
            if base_weight.squeeze().ndim < 2:
                continue
            base_2d_repr, trans, shp = basis.get_2d_repr(base_weight)
            u, s, vh = torch.linalg.svd(base_2d_repr, full_matrices=False)
            # need buffer for u, s, and vh comms
            num_to_send = int(s.shape[0] * percent_to_send)
            send_u = u[:num_to_send].contiguous()
            send_s = s[:num_to_send].contiguous()
            send_vh = vh[:, :num_to_send].contiguous()

            recv_u, recv_s, recv_vh = torch.zeros_like(send_u), torch.zeros_like(send_s), torch.zeros_like(send_vh)

            send_u_op = dist.P2POp(dist.isend, send_u, partner, tag=tag)
            recv_u_op = dist.P2POp(dist.irecv, recv_u, partner, tag=tag + 1)
            send_s_op = dist.P2POp(dist.isend, send_s, partner, tag=tag + 2)
            recv_s_op = dist.P2POp(dist.irecv, recv_s, partner, tag=tag + 3)
            send_vh_op = dist.P2POp(dist.isend, send_vh, partner, tag=tag + 4)
            recv_vh_op = dist.P2POp(dist.irecv, recv_vh, partner, tag=tag + 5)
            tag += 6
            waits = dist.batch_isend_irecv([send_u_op, recv_u_op, send_s_op, recv_s_op, send_vh_op, recv_vh_op])
            # c = 0
            for w in waits:
                w.wait()
                # c += 1
            # print(c)
            assert torch.all(recv_vh != 0.0), "No sending..."

            # my_usvh[n] = {'u': u, 's': s, 'vh': vh, "trans": trans, "shp": shp, "waits": waits}
            # TODO: best way to distribute the workload here? 2 for loops to allow comes to finish?
            #    lets just make it blocking for now, uncomment later to test (also need to use self.my_usvh)
            # for n in self.names_to_cat1d:

            # u
            # waits[0].wait()
            # waits[1].wait()
            u[-num_to_send:] = recv_u
            # waits[2].wait()
            # waits[3].wait()
            s[-num_to_send:] = recv_s
            # waits[4].wait()
            # waits[5].wait()
            vh[:, -num_to_send:] = recv_vh
            # undo 1d concatenating
            base_2d_repr = u @ s.diag() @ vh
            # set ND param
            if trans:
                base_2d_repr = base_2d_repr.T
            base_2d_repr = base_2d_repr.view(shp)
            base_weight.zero_()
            base_weight.add_(base_2d_repr)
            if self.comm_reset_opt_on_sync:
                utils.reset_adam_state(self.optimizer, base_weight)

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
    def _all_to_all_sync_cat1d(self):
        # in this sync, all procs give them same number of ranks, last ones are set to zero and ignored??

        # TODO: check memory hit, might be very high if keeping all the USVh mats
        for n in self.names_to_cat1d:
            base_weight = utils.rgetattr(self.model, n)
            base_2d_repr, trans, shp = basis.get_2d_repr(base_weight)
            catted, cat_names_n = basis.get_2d_rep_w_1d_names(
                model=self.model,
                base_2d=base_2d_repr,
                base_name=n,
                names1d=self.names_to_cat1d,
            )
            self.cat1d_dims[n] = cat_names_n
            u, s, vh = torch.linalg.svd(catted, full_matrices=False)
            # this one will use USVh in place and will just move things around
            # leftover = s.shape[0] % self.world_size -> handled automatically by zeroing
            keep_u = u.shape[0] // self.world_size
            st_u, sp_u = keep_u * self.rank, keep_u * (self.rank + 1)
            keep_s = s.shape[0] // self.world_size
            st_s, sp_s = keep_s * self.rank, keep_s * (self.rank + 1)
            keep_vh = vh.shape[1] // self.world_size
            st_vh, sp_vh = keep_vh * self.rank, keep_vh * (self.rank + 1)
            u_sel = u[:keep_u].clone()
            s_sel = s[:keep_s].clone()
            vh_sel = vh[:, :keep_vh].clone()
            u.zero_()
            s.zero_()
            vh.zero_()
            u[st_u:sp_u] += u_sel
            s[st_s:sp_s] += s_sel
            vh[:, st_vh:sp_vh] += vh_sel
            # if self.rank == 0:
            #     print(f"{n} u: {u.shape} {keep_u}, s: {s.shape} {keep_s}, vh: {vh.shape} {keep_vh}")

            u = u.contiguous()
            s = s.contiguous()
            vh = vh.contiguous()

            wait_u = dist.all_reduce(u, async_op=True)
            wait_s = dist.all_reduce(s, async_op=True)
            wait_vh = dist.all_reduce(vh, async_op=True)
            # TODO: seperate loop to let comms have time????
            wait_u.wait()
            wait_s.wait()
            wait_vh.wait()
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
                if self.comm_reset_opt_on_sync:
                    utils.reset_adam_state(self.optimizer, param)
            # set ND param
            if trans:
                base_2d_repr = base_2d_repr.T
            base_2d_repr = base_2d_repr.view(shp)
            base_weight.zero_()
            base_weight.add_(base_2d_repr)
            if self.comm_reset_opt_on_sync:
                utils.reset_adam_state(self.optimizer, base_2d_repr)

    @torch.no_grad()
    def _all_to_all_sync_no_cat1d(self):
        # in this sync, all procs give them same number of ranks, last ones are set to zero and ignored??

        # TODO: check memory hit, might be very high if keeping all the USVh mats
        for n, base_weight in self.model.named_parameters():
            if base_weight.squeeze().ndim < 2:
                continue
            base_2d_repr, trans, shp = basis.get_2d_repr(base_weight)
            u, s, vh = torch.linalg.svd(base_2d_repr, full_matrices=False)
            # this one will use USVh in place and will just move things around
            keep_u = u.shape[0] // self.world_size
            st_u, sp_u = keep_u * self.rank, keep_u * (self.rank + 1)
            keep_s = s.shape[0] // self.world_size
            st_s, sp_s = keep_s * self.rank, keep_s * (self.rank + 1)
            keep_vh = vh.shape[1] // self.world_size
            st_vh, sp_vh = keep_vh * self.rank, keep_vh * (self.rank + 1)
            u_sel = u[:keep_u].clone()
            s_sel = s[:keep_s].clone()
            vh_sel = vh[:, :keep_vh].clone()
            u.zero_()
            s.zero_()
            vh.zero_()
            u[st_u:sp_u] += u_sel
            s[st_s:sp_s] += s_sel
            vh[:, st_vh:sp_vh] += vh_sel

            u = u.contiguous()
            s = s.contiguous()
            vh = vh.contiguous()

            wait_u = dist.all_reduce(u, async_op=True)
            wait_s = dist.all_reduce(s, async_op=True)
            wait_vh = dist.all_reduce(vh, async_op=True)
            # TODO: seperate loop to let comms have time????
            wait_u.wait()
            wait_s.wait()
            wait_vh.wait()
            # undo 1d concatenating
            base_2d_repr = u @ s.diag() @ vh
            # set ND param
            if trans:
                base_2d_repr = base_2d_repr.T
            base_2d_repr = base_2d_repr.view(shp)
            base_weight.zero_()
            base_weight.add_(base_2d_repr)
            if self.comm_reset_opt_on_sync:
                utils.reset_adam_state(self.optimizer, base_weight)

    @torch.no_grad()
    def _all_to_all_sync(self):
        waits = []
        for n in self.names_to_average:
            p = utils.rgetattr(self.model, n)
            p /= self.world_size
            waits.append(dist.all_reduce(p, async_op=True))

        if self.cat1d:
            self._all_to_all_sync_cat1d()
        else:
            self._all_to_all_sync_no_cat1d()

        for w in waits:
            w.wait()

    @torch.no_grad()
    def _one_to_all_sync_cat1d(self, root: int, percent_to_send: float):
        # send a percentage of one rank to all the others

        # trading is on dim0 for U and V (making it dim1 for Vh)

        # steps:
        #   1. get 2d reps (if 1d, go through that)
        #   2. get amount to send + send nonblocking, U will take longest, but with all the SVDs, it probably doesnt matter
        #   3. recv from partner, create new USVh
        #   4. set new USVh to model
        if percent_to_send > 1:
            raise ValueError(f"Percent to send must be < 1 -> {percent_to_send}")

        # TODO: check memory hit, might be very high if keeping all the USVh mats
        for n in self.names_to_cat1d:
            base_weight = utils.rgetattr(self.model, n)
            base_2d_repr, trans, shp = basis.get_2d_repr(base_weight)
            catted, cat_names_n = basis.get_2d_rep_w_1d_names(
                model=self.model,
                base_2d=base_2d_repr,
                base_name=n,
                names1d=self.names_to_cat1d,
            )
            self.cat1d_dims[n] = cat_names_n
            u, s, vh = torch.linalg.svd(catted, full_matrices=False)
            # need buffer for u, s, and vh comms
            u_sel = u[: int(u.shape[0] * percent_to_send)].contiguous()
            s_sel = s[: int(s.shape[0] * percent_to_send)].contiguous()
            vh_sel = vh[:, : int(vh.shape[1] * percent_to_send)].contiguous()

            if self.rank != root:
                u_sel.zero_()
                s_sel.zero_()
                vh_sel.zero_()

            wait_u = dist.broadcast(u_sel, src=root, async_op=True)
            wait_s = dist.broadcast(s_sel, src=root, async_op=True)
            wait_vh = dist.broadcast(vh_sel, src=root, async_op=True)
            # TODO: seperate loop to let comms have time????
            wait_u.wait()
            wait_s.wait()
            wait_vh.wait()

            if self.rank != root:
                u[-int(u.shape[0] * percent_to_send) :] = u_sel
                s[-int(s.shape[0] * percent_to_send) :] = s_sel
                # vh[:, -int(vh.shape[0] * percent_to_send):] = vh_sel

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
                if self.comm_reset_opt_on_sync:
                    utils.reset_adam_state(self.optimizer, param)
            # set ND param
            if trans:
                base_2d_repr = base_2d_repr.T
            base_2d_repr = base_2d_repr.view(shp)
            base_weight.zero_()
            base_weight.add_(base_2d_repr)
            if self.comm_reset_opt_on_sync:
                utils.reset_adam_state(self.optimizer, base_2d_repr)

    @torch.no_grad()
    def _one_to_all_sync_no_cat1d(self, root: int, percent_to_send: float):
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
        for n, base_weight in self.model.named_parameters():
            if base_weight.ndim < 2:
                continue
            base_2d_repr, trans, shp = basis.get_2d_repr(base_weight)
            u, s, vh = torch.linalg.svd(base_2d_repr, full_matrices=False)
            # need buffer for u, s, and vh comms

            u_sel = u[: int(u.shape[0] * percent_to_send)].contiguous()
            s_sel = s[: int(s.shape[0] * percent_to_send)].contiguous()
            vh_sel = vh[:, : int(vh.shape[1] * percent_to_send)].contiguous()

            if self.rank != root:
                u_sel.zero_()
                s_sel.zero_()
                vh_sel.zero_()

            wait_u = dist.broadcast(u_sel, src=root, async_op=True)
            wait_s = dist.broadcast(s_sel, src=root, async_op=True)
            wait_vh = dist.broadcast(vh_sel, src=root, async_op=True)
            # TODO: seperate loop to let comms have time????
            wait_u.wait()
            wait_s.wait()
            wait_vh.wait()

            if self.rank != root:
                u[-int(u.shape[0] * percent_to_send) :] = u_sel
                s[-int(s.shape[0] * percent_to_send) :] = s_sel
                vh[:, -int(vh.shape[0] * percent_to_send) :] = vh_sel
            # undo 1d concatenating
            base_2d_repr = u @ s.diag() @ vh
            # set ND param
            if trans:
                base_2d_repr = base_2d_repr.T
            base_2d_repr = base_2d_repr.view(shp)
            base_weight.zero_()
            base_weight.add_(base_2d_repr)
            if self.comm_reset_opt_on_sync:
                utils.reset_adam_state(self.optimizer, base_weight)

    @torch.no_grad()
    def _one_to_all_sync(self, root: int, percent_to_send: float):
        waits = []
        for n in self.names_to_average:
            p = utils.rgetattr(self.model, n)
            p /= self.world_size
            waits.append(dist.all_reduce(p, async_op=True))

        if self.cat1d:
            self._one_to_all_sync_cat1d(root=root, percent_to_send=percent_to_send)
        else:
            self._one_to_all_sync_no_cat1d(root=root, percent_to_send=percent_to_send)

        for w in waits:
            w.wait()

    def _pre_forward(self):
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

    @torch.no_grad()
    def _post_train_step(self):
        # def _pre_forward(self):
        # select model for forward step
        # if self.total_train_iterations < self.warmup_steps:
        #     self.model_to_run = self.models['global']
        # print(self.total_train_iterations, self.warmup_steps)
        if self.total_train_iterations == self.warmup_steps and not self.no_ddp:
            if self.rank == self.logging_rank:
                log.info(
                    f"End of Warmup: {self.total_train_iterations} current epoch: {self.current_iter}/{self.iterations_per_train}",
                )
            # if warmup is done, transition to individual mode
            # params = {}
            # for n, p in self.model_to_run.named_parameters():
            #     params[n] = p.clone().detach()
            if isinstance(self.model_to_run, DDP):
                self.model_to_run = self.model_to_run.module
            self.model_to_run._ddp_params_and_buffers_to_ignore = self.ddp_params_and_buffers_to_ignore
            # if self.rank == self.logging_rank:
            #     print(self.ddp_params_and_buffers_to_ignore, type(self.model_to_run))

            # self.model_to_run = DDP(self.model) if len(self.names_to_always_sync) > 0 else self.model
            # self.model._ddp_params_and_buffers_to_ignore = self.names_to_not_average

            self.model_to_run = DDP(self.model_to_run)

            if self.comm_reset_opt_on_sync:
                for p in self.model_to_run.parameters():
                    utils.reset_adam_state(self.optimizer, p)
            if self.reset_lr_on_sync:
                self._setup_lr_rebound()

            # if self.rank == self.logging_rank:
            #     print(self.model_to_run)
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

                # pass
                # else:
                #     # if p.ndim == 1:
                #     #     hld = p
                #     #     trans = False
                #     #     shp = p.shape
                #     # else:
                #     hld, trans, shp = basis.get_2d_repr(p)
                #     hld = hld.contiguous()
                #     dist.all_reduce(hld, op=dist.ReduceOp.SUM)  # defualt op is sum
                #     if trans:
                #         hld = hld.T
                #     hld = hld.view(shp)
                #     p.zero_()
                #     p.add_(hld)
                if self.comm_reset_opt_on_sync:
                    utils.reset_adam_state(self.optimizer, p)

            if isinstance(self.model_to_run, DDP):
                self.model_to_run = self.model_to_run.module
            self.model_to_run._ddp_params_and_buffers_to_ignore = []
            self.model_to_run = DDP(self.model_to_run)

            if self.reset_lr_on_sync:
                self._setup_lr_rebound()

            # if self.rank == self.logging_rank:
            #     print(self.model_to_run)

            # self.model_to_run = self.model
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
                # if n in self.ddp_params_and_buffers_to_ignore or p.ndim == 1:
                #     pass  # pass not continue for opt param reset
                # else:
                #     # # if p.ndim == 1:
                #     # #     two_d_repr = p
                #     # #     trans = False
                #     # #     shp = p.shape
                #     # # else:
                #     two_d_repr, trans, shp = basis.get_2d_repr(p)

                #     # # print(f"{n} {start}, {stop}")
                #     # # if self.rank == self.logging_rank:
                #     # #     log.info(f"Removing weight rows: {n} {trans} {two_d_repr.shape}")
                #     # prune_dim = 1 if trans else 0
                #     # start = int((two_d_repr.shape[prune_dim] /self.world_size) * self.rank)
                #     # stop = int((two_d_repr.shape[prune_dim] /self.world_size) * (self.rank + 1))
                #     # prune_mask = get_prune_mask_from_2d_repr(
                #     #     two_d_repr, trans=trans, shp=shp, start=start, stop=stop, dim=prune_dim,
                #     # )
                #     # prune.custom_from_mask(rgetattr(self.model_to_run, module_n), 'weight', mask=prune_mask)

                #     # hld = two_d_repr
                #     # # hld *= prune_mask
                #     # if trans:
                #     #     hld = hld.T
                #     # hld = hld.view(shp)
                #     # hld *= prune_mask.to(hld.dtype)

                #     # # if self.rank == self.logging_rank:
                #     # #     print(two_d_repr[:10, :10])
                #     # # # p.zero_()

                #     # if self.rank == self.logging_rank:
                #     #     print(p[:10, :10])
                #     # p.add_(hld)

                #     u, s, vh = torch.linalg.svd(two_d_repr, full_matrices=False)
                #     if self.rank == self.logging_rank:
                #         # print(two_d_repr[:10, :10])
                #         nz10perc = torch.count_nonzero(s < s[0] * 0.1) / s.shape[0]
                #         nz1perc = torch.count_nonzero(s < s[0] * 0.01) / s.shape[0]
                #         nz50perc = torch.count_nonzero(s < s[0] * 0.5) / s.shape[0]

                #         print(
                #             f"{n}: {s.mean():.4f}, {s.min():.4f}, {s.max():.4f} "
                #             f"-- <10% of first: {nz10perc:.4f} <1% of first: {nz1perc:.4f} "
                #             f"<50% of first: {nz50perc:.4f}",
                #         )
                # -----------------------------------------

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
