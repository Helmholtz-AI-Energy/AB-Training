import time

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


class BasicTrainer:
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
        metrics: MetricCollection = None,
        iterations_per_train: int = None,
        log_freq: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_autocast = use_autocast
        self.scaler = GradScaler(enabled=use_autocast)
        self.max_grad_norm = max_grad_norm
        self.infloader = CycleDataLoader(train_loader)
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.total_train_iterations, self.current_iter = 0, 0
        self.iterations_per_train = len(train_loader) if iterations_per_train is None else iterations_per_train
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
        # log_message = f"{self.current_iter}/{self.iterations_per_train}: "
        # for m in self.metrics:  # assume metrics is dict and from torchmetrics
        #     if m.startswith("Multiclass"):
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
            for n, p in self.model_to_run.named_parameters():
                print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
            raise ValueError("NaN loss")

        try:
            if self.metrics is not None:
                self.metrics(outputs, labels)
        except BaseException:
            # unclear what to do here :/
            pass

        self.scaler.scale(loss).backward()

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
        self.num_itters += 1
        self.lr_scheduler.step_update(num_updates=self.num_itters, metric=loss)

        return loss.item()

    def train(self) -> None:
        # NOTE: itterations subercedes num_epochs
        self.model_to_run.train()  # Put model in training mode

        self.current_iter = 1
        # t0 = time.perf_counter()
        for data in self.infloader:
            # data_time = time.perf_counter() - t0
            # t0 = time.perf_counter()
            loss = self._train_step(data)
            # batch_time = time.perf_counter() - t0
            # Logging, progress tracking, etc. can be added here
            # to be logged: loss, data_time, batch_time, metrics??
            self._log_train(loss)

            if self.current_iter == self.iterations_per_train:
                break
            self.current_iter += 1
            self.total_train_iterations += 1
        if self.metrics is not None:
            self.metrics.compute()

        return self.metrics

    def pre_validate(self) -> None:
        # Pre-validation logic (optional)
        pass


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
