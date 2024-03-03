import logging
import time

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

log = logging.getLogger(__name__)


class BasicTrainer(object):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        lr_scheduler=None,
        use_autocast: bool = False,
        max_grad_norm: float = 0.0,
        metrics=None,
        iterations_per_train: int = 10,
        max_train_iters: int = 100,
        log_freq: int = 20,
        logging_rank: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_autocast = use_autocast
        self.scaler = GradScaler(enabled=self.use_autocast)
        self.max_grad_norm = max_grad_norm
        self.infloader = CycleDataLoader(train_loader)
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.total_train_iterations = 0
        self.current_iter = 0
        self.lr_updates = 0
        self.max_train_iters = max_train_iters
        if iterations_per_train is None:
            log.info(f"No iterations per train specified, using len(train_loader): {len(train_loader)}")
            self.iterations_per_train = len(train_loader)
        else:
            self.iterations_per_train = iterations_per_train

        if dist.is_initialized():
            self.rank = dist.get_global_rank()
            self.world_size = dist.get_world_size()
            self.logging_rank = logging_rank
        else:
            self.rank, self.logging_rank = 0, 0
            self.world_size = 1
        self.log_freq = log_freq

        self.model_to_run = self.model

    def _pre_forward(self):
        pass

    def _pre_backward(self):
        pass

    def _pre_optimizer(self):
        pass

    def _pre_lr_scheduler(self):
        pass

    def _log_train(self, loss):
        pass

    def _post_train_step(self):
        pass

    def _train_step(self, data: tuple[torch.Tensor, torch.Tensor]) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        inputs, labels = data
        inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

        # Pre-forward (optional)
        self._pre_forward()

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_autocast):
            # NOTE: some things dont play well with autocast - one should not put anything aside from the model in here
            outputs = self.model_to_run(inputs)
            loss = self.criterion(outputs, labels)

        if torch.isnan(loss):
            if self.rank == self.logging_rank:
                for n, p in self.model_to_run.named_parameters():
                    print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
            raise ValueError("NaN loss in training")

        try:
            if self.metrics is not None:
                # NOTE: this will only work for MY SPECIFIC UC
                #       for other cases, write your own things here!!
                self.metrics(outputs, labels, loss, inputs.size(0))
        except BaseException:
            pass

        # Pre-backward (optional)
        self._pre_backward()

        # Backward pass
        self.scaler.scale(loss).backward()
        if self.max_grad_norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model_to_run.parameters(), self.max_grad_norm)

        # Pre-optimizer (optional)
        self._pre_optimizer()

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # Post-step (optional)
        self._pre_lr_scheduler()

        # LR scheduler
        self.total_train_iterations += 1
        self.current_iter += 1

        self.lr_updates += 1
        self.lr_scheduler.step_update(num_updates=self.lr_updates, metric=loss)

        return loss.item()

    def train(self) -> None:
        self.model_to_run.train()  # Put model in training mode

        self.current_iter = 0
        t00 = time.perf_counter()
        t0 = time.perf_counter()
        for data in self.infloader:
            data_time = time.perf_counter() - t0
            loss = self._train_step(data)
            batch_time = time.perf_counter() - t0

            try:
                self.metrics.batch_time.update(batch_time)
                self.metrics.data_time.update(data_time)
            except AttributeError:
                pass
            # Logging, progress tracking, etc. can be added here
            # to be logged: loss, data_time, batch_time, metrics??
            if self.current_iter % self.log_freq == 0:
                self._log_train(loss)

            self._post_train_step()
            # if self.logging_rank == self.rank:
            #     print(f"{self.current_iter}, {self.iterations_per_train}, {self.total_train_iterations}")
            if self.current_iter == self.iterations_per_train:
                break
            t0 = time.perf_counter()

        if self.current_iter % self.log_freq != 0:
            self._log_train(loss)
        if self.metrics is not None:
            return self.metrics.compute(), time.perf_counter() - t00
        return None, time.perf_counter() - t00

    def pre_validate(self) -> None:
        # Pre-validation logic (optional)
        pass

    def done_training(self):
        return self.current_iter >= self.max_train_iters


class CycleDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterator)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:  # End of dataloader reached
            self.iterator = iter(self.dataloader)  # Reset iterator
            batch = next(self.iterator)
        return batch
