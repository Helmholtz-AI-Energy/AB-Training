import logging
import time
from copy import deepcopy
from typing import cast

import omegaconf
import scipy
import torch
import torch.distributed as dist
import torch.nn.utils.prune as prune
from omegaconf import OmegaConf, open_dict
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from ..models.ab_lowrank_model import utils as ab_utils
from ..optimizers import utils as opt_utils
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
        tracker=None,
        already_converted_model=False,
    ):
        if not isinstance(config, OmegaConf):
            config = self.sanitize_config(config)
            config = OmegaConf.create(config)
        try:
            autocast = config.model.autocast
        except AttributeError:
            autocast = None
        try:
            grad_norm = config.training.max_grad_norm
        except AttributeError:
            grad_norm = None
        try:
            iterations_per_train = config.training.iterations_per_train
        except AttributeError:
            iterations_per_train = None
        try:
            log_freq = config.training.print_freq
        except AttributeError:
            log_freq = None
        try:
            log_rank = config.tracking.logging_rank
        except AttributeError:
            log_rank = None

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            use_autocast=autocast,
            max_grad_norm=grad_norm,
            iterations_per_train=iterations_per_train,
            log_freq=log_freq,
            logging_rank=log_rank,
        )

        self.config = config
        self.warmup_steps = config.training.ab.warmup_steps
        self.steps_btw_syncing = config.training.ab.steps_btw_syncing
        self.ddp_steps_after_sync = config.training.ab.ddp_steps_after_sync
        self.reset_lr_on_sync = config.training.ab.reset_lr_on_sync
        # ---------------- MISC THINGS ---------------------------------------------------------
        self.comm_generator = torch.Generator().manual_seed(123456)  # no device -> use cpu
        if isinstance(self.model, DDP):  # get local model is the model is already DDP, will reinit with my stuff anyway
            self.model = self.model.module
        self.local_model = self.model
        self.tracker = tracker
        # --------------------------------------------------------------------------------------
        # ------------------------ SYNC / COMM MODES -------------------------------------
        self.sync_mode = config.training.ab.sync_mode
        if self.sync_mode not in ["full", "with-stale", "only-learned", "only-b"]:
            raise ValueError(
                f"sync mode must be one of [full, with-stale, only-learned, only-b], current: {self.sync_mode}",
            )
        # for more info on how syncing works, see self._sync_processes

        # TODO: clean me up...
        self.ab_train_comm_setup = config.training.ab.train_comm_setup
        if self.ab_train_comm_setup not in ["individual", "groups"]:
            raise ValueError(f"AB train mode must be one of [individual, groups], current: {self.ab_train_comm_setup}")

        if config.training.ab.train_comm_setup == "groups":
            self.use_groups = True
        else:
            self.use_groups = False

        if self.use_groups and self.sync_mode == "only-b" and self.im_logging_rank:
            log.info("Using 'only-b' syncing and groups is synchronous DDP on only B (Simga @ Vt)")

        # create the sub-groups in all cases, doesnt hurt (TODO: check that...)
        all_ranks = list(range(0, dist.get_world_size()))
        if config.training.ab.group_size == -1:
            group_ranks = [all_ranks[0 : len(all_ranks) // 2], all_ranks[len(all_ranks) // 2 :]]
            num_groups = 2
        else:
            group_size = int(config.training.ab.group_size)
            # increase group size if needed
            num_groups = self.world_size // group_size
            num_rem_ranks = self.world_size % group_size
            # print(num_groups, num_rem_ranks, self.world_size, group_size)
            group_sizes = [group_size] * num_groups
            for r in range(num_rem_ranks):
                group_sizes[r] += 1
            group_ranks = []
            for r in range(num_groups):
                group_ranks.append(all_ranks[sum(group_sizes[:r]) : sum(group_sizes[: r + 1])])

        # Create two new groups of equal size
        if not self.sync_mode == "only-b":
            self.num_a_groups = len(group_ranks[: num_groups // 2])
            self.num_b_groups = len(group_ranks[num_groups // 2 :])
            a_groups = torch.tensor(group_ranks[: num_groups // 2])
            b_groups = torch.tensor(group_ranks[num_groups // 2 :])

            log.info(f"Groups:\na: {group_ranks[: num_groups // 2]}\nb: {group_ranks[num_groups // 2:]}")

            self.group_a_ranks = a_groups.flatten().tolist()
            self.group_b_ranks = b_groups.flatten().tolist()
            local_ab_group, self.a_groups = dist.new_subgroups_by_enumeration(a_groups)
            if local_ab_group is not None:
                self.my_ab_group = local_ab_group
                self.my_train_ab_mode = "a"

            local_ab_group, self.b_groups = dist.new_subgroups_by_enumeration(b_groups)
            if local_ab_group is not None:
                self.my_ab_group = local_ab_group
                self.my_train_ab_mode = "b"

            self.full_a_group = dist.new_group(self.group_a_ranks, use_local_synchronization=True)
            self.full_b_group = dist.new_group(self.group_b_ranks, use_local_synchronization=True)

            # print(self.my_ab_group)
            test = torch.tensor([1], device=self.device)
            dist.all_reduce(test, group=self.my_ab_group)
            dist.all_reduce(test, group=self.full_a_group)
            dist.all_reduce(test, group=self.full_b_group)
            # print(f"testing group: {group_sizes} test allreduce: {test}")

        else:
            log.info(f"Groups:\na: None\nb: {group_ranks}")
            self.num_a_groups = 0
            self.num_b_groups = len(group_ranks)
            self.group_a_ranks = []

            b_groups = group_ranks
            self.group_b_ranks = all_ranks
            self.a_groups = []
            self.b_groups = []
            for gr in b_groups:
                self.b_groups.append(dist.new_group(ranks=gr, use_local_synchronization=True))
                if self.rank in gr:
                    self.my_ab_group = self.b_groups[-1]
                    self.my_train_ab_mode = "b"
            self.full_a_group = None
            self.full_b_group = None  # both are the full normal world

        if "full_rank_sync_name" not in config.training.ab or config.training.ab.full_rank_sync_names is None:
            full_rank_sync_names = []
        else:
            full_rank_sync_names = config.training.ab.full_rank_sync_names

        # names to always sync -> sync every step
        self.full_rank_sync_names = full_rank_sync_names
        self.all_names = []
        # names to not sync during individual training
        # self.ddp_params_and_buffers_to_ignore = []

        if config.training.ab.sync_mode == "only-b":
            if self.im_logging_rank:
                log.info("Setting split sigma to False for only-b training")
            with open_dict(config):
                config.training.ab.split_sigma = False
        # Create a list of modules to ignore from the specified weights
        skip_modules = []
        for n in self.full_rank_sync_names:
            mod_name = n.split(".")[:-1]
            layer_name = ".".join(mod_name)
            try:
                skip_modules.append(rgetattr(model, layer_name))
            except AttributeError:
                log.info(f"No module with name: {n} in model")

        if not already_converted_model:
            self.model = ab_utils.convert_network_ab_lowrank(self.model, config, skip_modules)
        self.model.train()

        # Get lists of param names: always sync in full, 1d, multi-dim weights, a, b
        self.param_name_lists = {"always-sync": [], "1d": [], "Nd": [], "a": [], "b": []}
        params_to_add_to_opt = []
        for n, p in self.model.named_parameters():
            if n in self.full_rank_sync_names or len(full_rank_sync_names) == 0:
                self.param_name_lists["always-sync"].append(n)
            elif p.squeeze().ndim == 1:
                self.param_name_lists["1d"].append(n)
            elif n.endswith(".a"):
                self.param_name_lists["a"].append(n)
                params_to_add_to_opt.append(p)
            elif n.endswith(".b"):
                self.param_name_lists["b"].append(n)
                params_to_add_to_opt.append(p)
            elif n.endswith((".weight", "full_rank_weight")):  # TODO: there should be a better rule here...
                # should be only weights with > 1D now, A/B will have different names
                # need to add another thing here to have the base_layer in the name
                self.param_name_lists["Nd"].append(n)
                params_to_add_to_opt.append(p)
            else:
                raise ValueError(f"name: {n} not accounted for in lists")
        # ------------------------------- ADD A/B TO OPTIMIZER ----------------------------
        # remove old param group, then add a new one with all the params
        # assuming that there is only one params group.....
        # taken from Optimizer.init
        params = self.model.parameters()
        self.optimizer.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.optimizer.add_param_group(cast(dict, param_group))
        # -------------------------------- DDP + WARMUP SETUP ------------------------------

        # start in warmup mode (standard global DDP)
        if "sync_batchnorm" in config.training:
            self.sync_bn = config.training.sync_batchnorm
        else:
            self.sync_bn = False

        if self.sync_bn:
            self.sbn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(device)
            self.model_to_run = self.sbn_model
        else:
            self.sbn_model = self.model.to(device)

        # print(f"warmup_steps: {self.warmup_steps}")
        if self.warmup_steps > 0:
            # starting in warmup phase
            templ = self.param_name_lists["a"] + self.param_name_lists["b"]
            self.sbn_model._ddp_params_and_buffers_to_ignore = templ
            ab_utils.change_ab_train_mode(  # TODO: test me, unsure if this will fail
                self.sbn_model,
                ab_training_mode="full",
                cut_singular_values=False,
            )
            ddp_model = DDP(self.sbn_model, process_group=None)
            self.current_train_mode = "full-rank"
        else:
            # no warmup, starting in AB training
            # TODO: set up logic for different syncing modes
            ab_utils.change_ab_train_mode(
                self.sbn_model,
                ab_training_mode=self.my_train_ab_mode,
                cut_singular_values=False,
            )

            ign = self.param_name_lists["Nd"] + self.param_name_lists["a" if self.my_train_ab_mode == "b" else "b"]
            self.sbn_model._ddp_params_and_buffers_to_ignore = ign
            ddp_model = DDP(self.sbn_model, process_group=self.my_ab_group)
            self.current_train_mode = "group"

        if self.warmup_steps > 0 or len(self.names_to_always_sync) != 0:
            self.model_to_run = ddp_model  # ddp on all ranks
        # ----------------------------------------------------------------------------------
        self.comm_reset_opt_on_sync = config.training.ab.reset_opt
        # ------------------------- LR Rebound ------------------------------
        self.in_lr_rebound = False
        self.lr_rebound_steps = config.training.ab.lr_rebound_steps
        self.current_lr_rebound_step = 0
        self.lr_rebound_step_factor = [0] * len(self.optimizer.param_groups)
        if hasattr(self.optimizer, "optimizer"):
            self.lr_rebound_step_factor = [0] * len(self.optimizer.optimizer.param_groups)
        self.warmup_lr = 1e-5
        # self.warmup_lr = config.training.lr_schedule.warmup_lr
        self.target_weight_decays = []
        for pg in self.optimizer.param_groups:
            if "weight_decay" in pg:
                self.target_weight_decays.append(pg["weight_decay"])
        # -------------------------------------------------------------------

    @staticmethod
    def sanitize_config(config):
        def filter_omegaconf_compatible(config_dict):
            """Removes keys with non-OmegaConf-compatible values from a dictionary.

            Args:
                config_dict: The dictionary to filter.

            Returns:
                A new dictionary containing only OmegaConf-compatible key-value pairs.
            """
            filtered_dict = {}
            for key, value in config_dict.items():
                if is_omegaconf_compatible(value):
                    filtered_dict[key] = value
            return filtered_dict

        def is_omegaconf_compatible(value):
            """Checks if a value is compatible for use in an OmegaConf DictConfig.

            Args:
                value: The value to test.

            Returns:
                True if the value is compatible, False otherwise.
            """

            try:
                # 1. Basic types compatibility
                if isinstance(value, (bool, int, float, str, type(None))):
                    return True

                # 2. Handling lists and dictionaries
                elif isinstance(value, (list, tuple)):  # OmegaConf supports lists and tuples
                    return all(is_omegaconf_compatible(item) for item in value)
                elif isinstance(value, dict):
                    return all(is_omegaconf_compatible(k) and is_omegaconf_compatible(v) for k, v in value.items())

                # 3. Attempt to create a temporary DictConfig (best effort)
                else:
                    omegaconf.DictConfig({"_temp": value})
                    return True

            except (omegaconf.errors.OmegaConfBaseException, TypeError):
                return False

        filtered_conf = filter_omegaconf_compatible(config)
        return filtered_conf

    def _prep_ab_train(self, cut_singular_values=False):
        # TODO: fix the DDP issue in the model/ddp model/local model.....im getting tired
        if isinstance(self.model_to_run, DDP):
            model = self.model_to_run.module
        else:
            model = self.model_to_run
        ab_utils.change_ab_train_mode(
            model,
            ab_training_mode=self.my_train_ab_mode,
            cut_singular_values=cut_singular_values,
        )
        # need to resize the optimizer states after change ab_train_mode
        if cut_singular_values:
            opt_utils.change_adam_shapes(self.optimizer)
            try:
                opt_utils.change_adam_shapes(self.optimizer.optimizer)
            except AttributeError:
                pass

        logmsg = f"AB Training: mode {self.my_train_ab_mode}\tDDP Ignore list includes: Nd, "
        nd = self.param_name_lists["Nd"]
        if self.ab_train_comm_setup == "groups":
            ab = "a" if self.my_train_ab_mode == "b" else "b"
            ign = nd + self.param_name_lists[ab]
            logmsg += f"{ab}, "
        elif self.ab_train_comm_setup == "individual":
            ign = nd + self.param_name_lists["a"] + self.param_name_lists["b"]
            logmsg += "A, B"
        else:
            raise ValueError(
                f"self.ab_train_comm_setup must be either 'groups' or 'individual': current {self.ab_train_comm_setup}",
            )

        if self.im_logging_rank:
            log.info(logmsg)

        model._ddp_params_and_buffers_to_ignore = ign
        dist.barrier(group=self.my_ab_group)
        self.model_to_run = DDP(model, process_group=self.my_ab_group)
        self.current_train_mode = "group"

    def _prep_full_rank_train(self):
        if isinstance(self.model_to_run, DDP):
            model = self.model_to_run.module
        else:
            model = self.model_to_run
        ab_utils.change_ab_train_mode(
            model,
            ab_training_mode="full",
            cut_singular_values=False,
        )
        ign = self.param_name_lists["a"] + self.param_name_lists["b"]
        if self.im_logging_rank:
            log.info("Full Rank Syncronous Training\tDDP Ignore list includes: A, B")
        model._ddp_params_and_buffers_to_ignore = ign
        self.model_to_run = DDP(model, process_group=None)
        self.current_train_mode = "full-rank"
        return self.model_to_run

    @torch.no_grad()
    def _sync_processes(self):
        # if in groups: need to average all non-AB ranks as well
        sync_time = time.perf_counter()
        waits = []
        mod = self.model_to_run if not isinstance(self.model_to_run, DDP) else self.model_to_run.module
        if self.ab_train_comm_setup == "groups" and self.sync_mode != "only-b":
            # need to average 1D and always sync here because they are only trained in the group
            names_to_avg_1d = self.param_name_lists["1d"] + self.param_name_lists["always-sync"]
            for n, p in mod.named_parameters():
                if n in names_to_avg_1d:
                    waits.append(dist.all_reduce(p, op=dist.ReduceOp.AVG, async_op=True))

        # three options: full, with-stale, and only-learned
        # full mode: -> transition network to 'full' mode, do average
        if self.sync_mode == "full":
            waits = self._full_sync(waits, mod)
        elif self.sync_mode == "with-stale":
            waits = self._stale_sync(waits, mod)
        elif self.sync_mode == "only-learned":
            waits = self._only_learned_sync(waits, mod)
        elif self.sync_mode == "only-b":
            waits = self._only_b_sync(waits, mod)
        if len(waits) == 0:
            if self.im_logging_rank:
                log.debug("No waits in outer 1D loop for sync_processes")
        for w in waits:
            if w is not None:
                w.wait()

        if self.im_logging_rank:
            log.info(
                f"Syncing processes. comm setup: {self.ab_train_comm_setup}. "
                f"sync mode: {self.sync_mode}. time needed: {time.perf_counter() - sync_time}",
            )
        return mod

    def _full_sync(self, waits: list, mod) -> tuple[list, torch.nn.Module]:
        # full mode: -> transition network to 'full' mode, do average
        ab_utils.change_ab_train_mode(
            self.model_to_run,
            ab_training_mode="full",
            cut_singular_values=False,
        )

        for n, p in mod.named_parameters():
            if n in self.param_name_lists["Nd"]:
                waits.append(dist.all_reduce(p, op=dist.ReduceOp.AVG, async_op=True))
        return waits

    def _stale_sync(self, waits: list, mod) -> tuple[list, torch.nn.Module]:
        # 'with-stale' mode: -> average all 'a' weights, average all 'b' weights, find full-rank repr
        ab = self.param_name_lists["a"] + self.param_name_lists["b"]
        ab_waits = []
        for n, p in mod.named_parameters():
            if n in ab:
                if not p.is_contiguous():
                    p.set_(p.contiguous())
                p.div_(self.world_size)
                ab_waits.append(dist.all_reduce(p, op=dist.ReduceOp.SUM, async_op=True))
        waits += ab_waits
        return waits

    def _only_learned_sync(self, waits: list, mod) -> tuple[list, torch.nn.Module]:
        # 'only-learned' mode ->
        #   'a' and 'b' are already the same within the groups, send them to the other group, find full-rank repr
        #       need to average A between all the A groups, and B with all the B groups
        ab_waits = []

        for n, p in mod.named_parameters():
            if n in self.param_name_lists["a"]:
                # average A across a-groups
                if self.num_a_groups > 1 and self.rank in self.group_a_ranks:
                    dist.all_reduce(p, op=dist.ReduceOp.AVG, group=self.full_a_group)
                root = self.group_a_ranks[0]
                # send a to b-groups/all
                if not p.is_contiguous():
                    with torch.no_grad():
                        p.set_(p.contiguous())
                ab_waits.append(dist.broadcast(p, src=root, async_op=True))
            # send b to a-groups/all
            if n in self.param_name_lists["b"]:
                # average B across b-groups
                if self.num_b_groups > 1 and self.rank in self.group_b_ranks:
                    dist.all_reduce(p, op=dist.ReduceOp.AVG, group=self.full_b_group)
                root = self.group_b_ranks[0]
                # send a to b-groups/all
                if not p.is_contiguous():
                    with torch.no_grad():
                        p.set_(p.contiguous())
                ab_waits.append(dist.broadcast(p, src=root, async_op=True))
        waits += ab_waits
        return waits

    def _only_b_sync(self, waits: list, mod) -> tuple[list, torch.nn.Module]:
        # 'with-stale' mode: -> average all 'a' weights, average all 'b' weights, find full-rank repr
        ab_waits = []
        for n, p in mod.named_parameters():
            if n in self.param_name_lists["b"]:
                ab_waits.append(dist.all_reduce(p, op=dist.ReduceOp.AVG, async_op=True))
        waits += ab_waits
        return waits

    def _setup_lr_rebound(self):
        self.in_lr_rebound = True
        self.current_lr_rebound_step = 0
        current_lr = 0
        for c, pg in enumerate(self.optimizer.param_groups):
            current_lr += pg["lr"]
            # set target_lr to current lr! will break the warmup when we get to the same level
            # self.lr_rebound_step_factor[c] = (target_lr - self.warmup_lr) / self.lr_rebound_steps
            self.lr_rebound_step_factor[c] = (current_lr - self.warmup_lr) / self.lr_rebound_steps
            pg["lr"] = self.warmup_lr
        if hasattr(self.optimizer, "optimizer"):
            for c, pg in enumerate(self.optimizer.optimizer.param_groups):
                current_lr += pg["lr"]
                # set target_lr to current lr! will break the warmup when we get to the same level
                self.lr_rebound_step_factor[c] = (current_lr - self.warmup_lr) / self.lr_rebound_steps
                pg["lr"] = self.warmup_lr

    def _pre_forward(self):  # OVERWRITE base class
        # if not self.model_to_run.training:
        #     return
        # ---------------------- LR Rebound ----------------------------------------
        if not self.in_lr_rebound:
            return
        # if not in reboud, act normally. otherwise, increase the lr by lr_rebound_factor
        self.current_lr_rebound_step += 1
        # exist_lr_warmup = False
        for c, pg in enumerate(self.optimizer.param_groups):
            # pg["lr"] = self.lr_rebound_step_factor[c] * self.current_lr_rebound_step
            new_lr = self.lr_rebound_step_factor[c] * self.current_lr_rebound_step
            # this CHANGES the LR at the top of the iteration, and will overwrite the LR scheduler
            if new_lr < pg["lr"] and self.logging_rank == self.rank and self.current_lr_rebound_step % 10 == 0:
                log.info(f"In LR rebound: actual LR: {new_lr:.6f}")
            elif new_lr >= pg["lr"]:
                continue
            pg["lr"] = new_lr
        if hasattr(self.optimizer, "optimizer"):
            for c, pg in enumerate(self.optimizer.optimizer.param_groups):
                # pg["lr"] = self.lr_rebound_step_factor[c] * self.current_lr_rebound_step
                new_lr = self.lr_rebound_step_factor[c] * self.current_lr_rebound_step
                # this CHANGES the LR at the top of the iteration, and will overwrite the LR scheduler
                if new_lr < pg["lr"] and self.logging_rank == self.rank and self.current_lr_rebound_step % 10 == 0:
                    log.info(f"In LR rebound: actual LR: {new_lr:.6f}")
                elif new_lr >= pg["lr"]:
                    continue
                pg["lr"] = new_lr

            # # turning off WD during rebound
            # if "weight_decay" in pg:
            #     pg["weight_decay"] *= 0

        if self.current_lr_rebound_step >= self.lr_rebound_steps:
            # for c, pg in enumerate(self.optimizer.param_groups):
            #     if "weight_decay" in pg:
            #         pg["weight_decay"] += self.target_weight_decays[c]
            self.in_lr_rebound = False
        # --------------------------------------------------------------------------

    @torch.no_grad()
    def _post_train_step(self):
        # TODO: Docs
        if self.total_train_iterations < self.warmup_steps:
            self.current_train_mode = "full-rank"
            return
        elif self.total_train_iterations == self.warmup_steps:
            if self.rank == self.logging_rank:
                log.info(
                    f"End of Warmup: {self.total_train_iterations} current epoch: {self.current_iter}/{self.iterations_per_train}",
                )
            # if warmup is done, transition to AB training mode
            self._prep_ab_train(cut_singular_values=False)

            if self.comm_reset_opt_on_sync:
                for _n, p in list(self.model_to_run.named_parameters()):
                    utils.reset_adam_state(self.optimizer, p)
                    try:
                        utils.reset_adam_state(self.optimizer.optimizer, p)
                    except AttributeError:
                        pass
            if self.reset_lr_on_sync:
                self._setup_lr_rebound()

            ab_utils.get_network_compression(self.model_to_run, tracker=self.tracker)

        elif (
            self.total_train_iterations % self.steps_btw_syncing == 0
            and self.total_train_iterations > self.warmup_steps
        ):
            # Sync processes and start DDP training for full-rank data
            if self.rank == self.logging_rank:
                log.info(
                    f"Sycning ranks - iteration: {self.total_train_iterations} "
                    f"current epoch: {self.current_iter}/{self.iterations_per_train} "
                    f"Method: {self.sync_mode}",
                )
            # ab_utils.compare_basis_ab(self.model_to_run)

            synced_model = self._sync_processes()
            self.model_to_run = synced_model
            self._prep_full_rank_train()

            # Reset optimizer
            if self.comm_reset_opt_on_sync:
                for _n, p in list(self.model_to_run.named_parameters()):
                    utils.reset_adam_state(self.optimizer, p)
                    try:
                        utils.reset_adam_state(self.optimizer.optimizer, p)
                    except AttributeError:
                        pass
            # start LR rebound
            if self.reset_lr_on_sync:
                self._setup_lr_rebound()
            ab_utils.get_network_compression(self.model_to_run, tracker=self.tracker)
        elif (
            self.total_train_iterations > self.steps_btw_syncing
            and self.total_train_iterations % self.steps_btw_syncing == self.ddp_steps_after_sync
            # self.total_train_iterations % (self.steps_btw_syncing // 2) == 0
            # and self.total_train_iterations > self.warmup_steps
        ):
            # finish average here -> receive everything - wait? then we dont need to do the weighted average...
            # for now, can just have this be blocking

            if self.rank == self.logging_rank:
                log.info(
                    f"SWITCH TO AB TRAINING\nCompare sigma distribution: {self.total_train_iterations} "
                    f"current epoch: {self.current_iter}/{self.iterations_per_train}",
                )

            # ab_utils.compare_basis_ab(self.model_to_run)

            self._prep_ab_train(cut_singular_values=True)

            if self.comm_reset_opt_on_sync:
                for _n, p in list(self.model_to_run.named_parameters()):
                    utils.reset_adam_state(self.optimizer, p)
                    try:
                        utils.reset_adam_state(self.optimizer.optimizer, p)
                    except AttributeError:
                        pass

            if self.reset_lr_on_sync:
                self._setup_lr_rebound()
            ab_utils.get_network_compression(self.model_to_run, tracker=self.tracker)

        if self.current_iter == self.iterations_per_train:
            ab_utils.get_network_compression(self.model_to_run, tracker=self.tracker)

    def _log_train(self, loss):
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
        log.info(prnt_str)
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
