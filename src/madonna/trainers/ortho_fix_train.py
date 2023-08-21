from __future__ import annotations

import logging
import os
import random
import shutil
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path

import hydra
import mlflow.pytorch

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
import omegaconf
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from torch.autograd.grad_mode import no_grad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torchmetrics import MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall

import madonna
import wandb


@no_grad()
def _group_tensors_by_device_and_dtype(tensorlistlist, with_indices=False):
    assert all(
        [not x or len(x) == len(tensorlistlist[0]) for x in tensorlistlist],
    ), "all specified tensorlists must match in length"
    per_device_and_dtype_tensors = defaultdict(
        lambda: [[] for _ in range(len(tensorlistlist) + (1 if with_indices else 0))],
    )
    for i, t in enumerate(tensorlistlist[0]):
        key = (t.device, t.dtype)
        for j in range(len(tensorlistlist)):
            # a tensorlist may be empty/None
            if tensorlistlist[j]:
                per_device_and_dtype_tensors[key][j].append(tensorlistlist[j][i])
        if with_indices:
            # tack on previous index
            per_device_and_dtype_tensors[key][j + 1].append(i)
    return per_device_and_dtype_tensors


def _has_foreach_support(tensors, device: torch.device) -> bool:
    if device.type not in ["cpu", "cuda"] or torch.jit.is_scripting():
        return False
    return all([t is None or type(t) == torch.Tensor for t in tensors])


best_acc1 = 0
log = logging.getLogger(__name__)


def main(config):  # noqa: C901
    if config.tracker == "wandb":
        wandb_run = madonna.utils.tracking.check_wandb_resume(config)
    if config.training.resume and not wandb_run:  # resume an interrupted run
        ckpt = config.training.checkpoint
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
    with omegaconf.open_dict(config):
        config.save_dir = madonna.utils.utils.increment_path(
            Path(config.training.checkpoint_out_root) / config.name,
        )  # increment run
    if config.rank == 0:
        log.info(f"save_dir: {config.save_dir}")

    save_dir = Path(config.save_dir)

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    log.info(f"out_dir (wdir): {wdir}")
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    # results_file = save_dir / 'results.txt'
    # f = open(results_file, 'w')
    wandb_logger = madonna.utils.tracking.WandbLogger(config) if config.rank == 0 and config.enable_tracking else None

    if config.seed is not None:
        cudnn.benchmark = True
        cudnn.deterministic = True
        # torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=5)
        # if dist.is_initialized():
        #     scale = dist.get_rank() % 4
        # else:
        #     scale = 0
        random.seed(int(config["seed"]))
        torch.manual_seed(int(config["seed"]))

    if config.cpu_training:
        gpu = None
        log.debug("NOT using GPUs!")
        device = torch.device("cpu")
    else:
        if dist.is_initialized():
            gpu = dist.get_rank() % torch.cuda.device_count()  # only 4 gpus/node
            log.debug(f"Using GPU: {gpu}")
        else:
            gpu = 0
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    # print('before get model')
    model = madonna.utils.get_model(config)
    if not config.cpu_training:
        model.cuda(gpu)
    # print('before model wrapping')
    if config.training.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # TODO: fix model repr
    if not config.baseline:
        model_hold = hydra.utils.instantiate(config.training.fixing_method)
        model = model_hold(model).to(device)
        log.info("using SVD model")
        # if dist.get_rank() == 0:
        #     print(model.model)
    elif dist.is_initialized():
        model = DDP(model)  # , device_ids=[config.rank])
        log.info("using DDP baseline model")
        # if dist.get_rank() == 0:
        #     print(model)
    else:
        log.info("using baseline 1 process model")
        # print(model)

    criterion = madonna.utils.get_criterion(config)
    optimizer = madonna.utils.get_optimizer(config, model, lr=config.training.lr)
    # if not config.baseline:
    #     madonna.models.utils.change_optimizer_group_for_svd(optimizer, model=model.model, config=config)
    #     # params = model.model.parameters()
    #     ddp_model = model.model
    # else:
    #     # params = model.parameters()
    #     ddp_model = model

    # set optimizer for references for SVD model to reset shapes of state params
    if not config.baseline:
        model.set_optimizer(optimizer)

    dset_dict = madonna.utils.datasets.get_dataset(config)
    train_loader, train_sampler = dset_dict["train"]["loader"], dset_dict["train"]["sampler"]
    val_loader = dset_dict["val"]["loader"]

    # sam_opt = madonna.optimizers.svd_sam.SAM(params=params, base_optimizer=optimizer, rho=0.05, adaptive=False)
    train_len = len(train_loader)
    scheduler, _ = madonna.utils.get_lr_schedules(config, optimizer, train_len)
    # optimizer = madonna.optimizers.MixedSVDOpt(config=config, model=model)
    # if not config.baseline:
    #     model.set_optimizer(optimizer)

    # optionally resume from a checkpoint
    # Reminder: when resuming from a single checkpoint, make sure to call init_model with
    start_epoch = config.training.start_epoch
    if config.training.checkpoint is not None:  # TODO: FIXME!!
        if os.path.isfile(config.training.checkpoint):
            print(f"=> loading checkpoint: {config.training.checkpoint}")
            if torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{gpu}"
                checkpoint = torch.load(config.training.checkpoint, map_location=loc)
            start_epoch = checkpoint["epoch"]

            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    config.training.checkpoint,
                    start_epoch,
                ),
            )
            # LRWarmupLinear.current_step = len(train_loader) * start_epoch
            # print(f"new warmup step: {LRWarmupLinear.current_step}")
            # if not config.baseline:
            #     optimizer.param_groups[-1]["lr"] = config.training.sigma_optimizer.min_lr
            # # optimizer should have the correct LR from the loading point
            if not config.baseline:
                if "next_stability_iteration" in checkpoint:
                    model.next_stability_iteration = checkpoint["next_stability_iteration"]
                if "call_count" in checkpoint:
                    model.call_count = checkpoint["call_count"]
            if scheduler is not None and start_epoch > 0:
                if True:  # args.sched_on_updates: FIXME
                    scheduler.step_update(start_epoch * len(train_loader))
                else:
                    scheduler.step(start_epoch)
        else:
            print(f"=> no checkpoint found at: {config.training.checkpoint}")

    # if start_epoch == 0:
    # set the learning rate to be based on the min_lr
    # groups = optimizer.param_groups
    # if not config.baseline:
    #     groups = groups[:-1]
    # for group in groups:
    #     group["lr"] = config.training.min_lr + (warmup_scheduler.add_value * start_epoch)
    # if not config.baseline:
    #     optimizer.param_groups[-1]["lr"] = config.training.sigma_optimizer.min_lr + (
    #         warmup_scheduler.sigma_add * start_epoch
    #     )

    # out_base = None
    # if "checkpoint_out_root" in config.training and config.training.checkpoint_out_root is not None:
    #     out_base = Path(config.training.checkpoint_out_root)
    #     model_name = config.model.name
    #     dataset_name = config.data.dataset
    #     out_base /= "baseline" if config.baseline else "svd"
    #     out_base = out_base / f"{model_name}-{dataset_name}" / f"bs-{config.data.local_batch_size * config.world_size}"
    #     out_base.mkdir(parents=True, exist_ok=True)
    #     log.info(f"Saving model to: {out_base}")

    # # if config['evaluate']:
    # #     validate(val_loader, dlrt_trainer, config)
    # #     return

    train_metrics = MetricCollection(
        [
            MulticlassF1Score(task="multiclass", num_classes=config.data.classes),
            MulticlassPrecision(task="multiclass", num_classes=config.data.classes),
            MulticlassRecall(task="multiclass", num_classes=config.data.classes),
        ],
    ).to(device)
    val_metrics = MetricCollection(
        [
            MulticlassF1Score(task="multiclass", num_classes=config.data.classes),
            MulticlassPrecision(task="multiclass", num_classes=config.data.classes),
            MulticlassRecall(task="multiclass", num_classes=config.data.classes),
        ],
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=config.model.autocast)
    rank = dist.get_rank() if dist.is_initialized() else 0

    refactory_warmup = None
    len_train = len(train_loader)
    best_fitness = 0.0
    for epoch in range(start_epoch, config.training["epochs"]):
        torch.cuda.reset_peak_memory_stats()
        if config["rank"] == 0:
            # console.rule(f"Begin epoch {epoch} LR: {optimizer.param_groups[0]['lr']}")
            lr_list, prnt_str = get_lrs(optimizer)
            log.info(f"Begin epoch {epoch} LRs: {prnt_str}")
            lrs = {"lr": lr_list[0]}
            # if not config.baseline:
            #     metrics["sigma_lr"] = lr_dict["sigma"]
            # mlflow.log_metrics(
            #     metrics=metrics,
            #     step=epoch,
            # )
            if config.enable_tracking:
                # wandb_logger.log(metrics, step=epoch * len_train)
                wandb_logger.log(lrs)
        if dist.is_initialized() and config.data.distributed_sample and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # with torch.no_grad():
        #     if not config.baseline and model.all_stable:
        #         for n, p in model.named_parameters():
        #             if n.endswith(".s") or n.endswith("_s"):
        #                 # # TODO: remove later if not working
        #                 # sdiag = torch.diag(self.s).clone()
        #                 sdiag = torch.diag(p)
        #                 # sdiag_diff1 = torch.diff(sdiag, n=1) * 0.001
        #                 # sdiag_diff2 = torch.diff(sdiag, n=2) * 0.0001
        #                 # # sdiag_diff3 = torch.diff(sdiag, n=3) * 0.001
        #                 # for i in range(p.shape[0] - 1):
        #                 #     p[i, i + 1] = sdiag_diff1[i]
        #                 #     if i < p.shape[0] - 2:
        #                 #         p[i, i + 2] = sdiag_diff2[i]

        #                 mask = torch.abs(p) <= 1e-7
        #                 # umask = torch.abs(p) > 1e-7

        #                 # print(f"{n} -> {torch.count_nonzero(mask)}")
        #                 p[mask] *= 0
        #                 p[mask] += 1e-1 * torch.rand_like(p[mask]) * sdiag.min()

        train_loss, last_loss, refactory_warmup = train(
            train_loader=train_loader,
            optimizer=optimizer,
            model=model,
            criterion=criterion,
            epoch=epoch,
            device=device,
            config=config,
            scaler=scaler,
            lr_scheduler=scheduler,
            # sigma_lrs={"sched": sigma_sched, "warmup": sigma_warmup},
            refactory_warmup=refactory_warmup,
            mixup=dset_dict["mixup"],
            wandb_logger=wandb_logger,
            metrics=train_metrics,
        )
        if (
            not config.baseline
            and epoch >= config.training.svd_epoch_delay
            and epoch % config.training.svd_epoch_freq == 0
        ):
            _ = model.model_stability_tracking(force=True)

        # if epoch * len(train_loader) >= warmup_steps or epoch == config.training.epochs - 1:  # epoch % 2 == 1
        # # one block ---------------------------------------------------------------------------------------------
        # if not config.baseline and epoch in fib_epochs:
        #     # try:
        #     # dist.barrier()
        #     # if rank == 0:
        #     #     for n, p in model.named_parameters():
        #     #         if n.endswith(("_s", ".s")) and p.grad is not None:
        #     #             print(p.grad.mean(), p.grad.min(), p.grad.max(), p.mean())
        #     # try:
        #     #     scaler.step(optimizer=optimizer.sigma_opt)
        #     #     scaler.update()
        #     # except AssertionError:
        #     #     # if this fails its most likely because the sigma values dont have any grads yet
        #     #     pass
        #     # optimizer.step(step_sigma=True, step_opt1=False)
        #     optimizer.zero_grad(set_to_none=True, reset_sigma=True)
        #     stabtime = time.perf_counter()
        #     reset_optimizer, all_layers_stable = model.test_basis_stability_all_layers()
        #     if all_layers_stable and not all_stable:
        #         # NOTE: this will only do something when it isnt baseline
        #         if rank == 0:
        #             log.info("Deleting the full rank weights from the base optimizer")
        #         optimizer.remove_full_rank_weights()
        #         all_stable = all_layers_stable
        #     if reset_optimizer:
        #         optimizer.reset_shapes_of_sigma(model)
        #     if rank == 0:
        #         log.info(f"Stability time: {time.perf_counter() - stabtime}")
        # # -------------------------------------------------------------------------------------------------------

        # if model.skip_stability:
        # for n, p in model.named_parameters():
        #     print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        # model.sync_models()
        # if epoch % 10 == 9 or epoch == config.training.epochs - 1:
        #     if dist.get_rank() == 0:
        #         for n, p in model.named_parameters():
        #             print(f"{n} {p.requires_grad} sparcity: {torch.count_nonzero(torch.abs(p) < 1e-5) / p.numel()}")

        if config.rank == 0:
            log.info(f"Average Training loss across process space: {train_loss}")
        # evaluate on validation set
        t1, val_loss = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            config=config,
            epoch=epoch,
            device=device,
            print_on_rank=config.rank == 0,
            pg=None,
            len_train=len_train,
            wandb_logger=wandb_logger,
            metrics=val_metrics,
        )

        if config.rank == 0:
            log.info(
                f"Average val loss across process space: {val_loss} " f"-> diff: {train_loss - val_loss}",
            )

        # log the percentage of params in use
        if config.rank == 0 and not config.baseline:
            if hasattr(model, "get_perc_params_all_layers"):
                perc, trainable, normal = model.get_perc_params_all_layers()
                if config.enable_tracking:
                    # mlflow.log_metric("perc_parmas", perc, step=epoch)
                    wandb_logger.log({"perc_parmas": perc})  # , step=(epoch + 1) * len_train)
                log.info(f"% params: {perc:.3f}% trainable: {trainable} full: {normal}")
                # model.track_interior_slices_mlflow(config, epoch)

        # log metrics
        train_metrics_epoch = train_metrics.compute()
        val_metrics_epoch = val_metrics.compute()

        # Save model
        if rank == 0 and config.enable_tracking:
            if t1 > best_fitness:
                best_fitness = t1

            # print(train_metrics_epoch)

            log_dict = {
                "train/f1": train_metrics_epoch["MulticlassF1Score"],
                "train/precision": train_metrics_epoch["MulticlassPrecision"],
                "train/recall": train_metrics_epoch["MulticlassRecall"],
                # "train/loss": train_loss,
                "val/f1": val_metrics_epoch["MulticlassF1Score"],
                "val/precision": val_metrics_epoch["MulticlassPrecision"],
                "val/recall": val_metrics_epoch["MulticlassRecall"],
                # "val/loss": val_loss,
            }
            log.info(
                f"Epoch end metrics: \n\t"
                f"train f1/prec/rec: {log_dict['train/f1']:.4f} / "
                f"{log_dict['train/precision']:.4f} / {log_dict['train/recall']:.4f}"
                f"\n\tval f1/prec/rec: {log_dict['val/f1']:.4f} / "
                f"{log_dict['val/precision']:.4f} / {log_dict['val/recall']:.4f}",
            )
            wandb_logger.log(log_dict)

            wandb_logger.end_epoch(best_result=best_fitness == t1)
            ckpt = {
                "epoch": epoch,
                "best_fitness": best_fitness,
                # 'training_results': results_file.read_text(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "wandb_id": wandb_logger.wandb_run.id if wandb_logger.wandb else None,
            }

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == t1:
                torch.save(ckpt, best)
            if (best_fitness == t1) and (epoch >= 200):
                torch.save(ckpt, wdir / "best_{:03d}.pt".format(epoch))
            if epoch == 0:
                torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
            elif ((epoch + 1) % 10) == 0:
                torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
            elif epoch >= (config.training.epochs - 5):
                torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
            if wandb_logger.wandb and config.enable_tracking:
                if (
                    (epoch + 1) % config.training.save_period == 0 and not epoch == config.training.epochs - 1
                ) and config.training.save_period != -1:
                    wandb_logger.log_model(
                        last.parent,
                        config,
                        epoch,
                        t1,
                        best_model=best_fitness == t1,
                    )
            del ckpt
        # set up next epoch after saving everything
        # wandb.log({"epoch": epoch}, step=(epoch + 1) * len_train)
        scheduler.step(epoch + 1, metric=val_loss)
        train_metrics.reset()
        val_metrics.reset()


def get_lrs(opt):
    out_lrs = []
    prnt_str = ""
    for group in opt.param_groups:
        out_lrs.append(group["lr"])
        prnt_str += f"group {len(out_lrs)}: lr {group['lr']:.6f}\t"
    return out_lrs, prnt_str


def train(
    train_loader,
    optimizer: madonna.optimizers.MixedSVDOpt,
    model,
    criterion,
    epoch,
    device,
    config,
    wandb_logger,
    metrics,
    lr_scheduler=None,
    scaler=None,
    log=log,
    sigma_lrs=None,
    refactory_warmup=None,
    mixup=None,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )
    # switch to train mode
    model.train()
    model_time = 0
    if refactory_warmup is None:
        refactory_warmup = {}

    end = time.time()
    lr_reset_steps = config.training.lr_reset_period
    # last_lr = 0
    steps_remaining = 0
    step_factors = None
    print_norms = []
    updates_per_epoch = len(train_loader)
    num_updates = epoch * updates_per_epoch
    for i, data in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        if hasattr(config.data, "dali") and config.data.dali:
            images = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        else:
            images = data[0]
            target = data[1]
        if mixup is not None:
            images, target = mixup(images, target)
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        t0 = time.perf_counter()
        # reset_lr_warmup = False
        # if not config.baseline:
        #     _ = model.model_stability_tracking()
        # if reset_lr_warmup:
        #     # reset the LR to a small value to avoid degredation in training
        #     # last_lr = optimizer.opt1.param_groups[0]["lr"]
        #     step_factors = (
        #         optimizer.opt1.param_groups[0]["lr"] / lr_reset_steps,
        #         optimizer.sigma_opt.param_groups[0]["lr"] / lr_reset_steps,
        #         optimizer.opt1.param_groups[0]["lr"],
        #         optimizer.sigma_opt.param_groups[0]["lr"],
        #     )
        #     optimizer.opt1.param_groups[0]["lr"] /= lr_reset_steps
        #     if len(optimizer.opt1.param_groups) > 1:
        #         optimizer.opt1.param_groups[1]["lr"] /= lr_reset_steps
        #     optimizer.sigma_opt.param_groups[0]["lr"] /= lr_reset_steps
        #     steps_remaining = lr_reset_steps - 1

        # if config.rank == 0 and i % 4 == 0:
        #     _, lrs = optimizer.get_lrs()
        #     print(f"LRS: {lrs}")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.model.autocast):
            # NOTE: some things dont play well with autocast - one should not put anything aside from the model in here
            output = model(images)
            loss = criterion(output, target)

        if torch.isnan(loss):
            for n, p in model.named_parameters():
                print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
            raise ValueError("NaN loss")
        scaler.scale(loss).backward()
        # md = model if config.baseline else model.model
        # TODO: move this back into the if block with scaling

        # grads = [p.grad for p in model.parameters() if p.grad is not None]
        # norm_type = 2.0
        # if len(grads) == 0:
        #     return torch.tensor(0.)
        # first_device = grads[0].device
        # grouped_grads = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]
        # foreach = None
        # norms = []
        # for ((device, _), [grads]) in grouped_grads.items():
        #     if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
        #         norms.extend(torch._foreach_norm(grads, norm_type))
        #     elif foreach:
        #         raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        #     else:
        #         norms.extend([torch.norm(g, norm_type) for g in grads])

        # total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
        # print_norms.append(f"{total_norm.item():.5f}")

        if config.training.max_grad_norm != -1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        # if i % 4 == 3 or i == len(train_loader) - 1:
        # TODO: running sigma?
        # running_sigma = False
        model_time += time.perf_counter() - t0

        # for n, p in model.named_parameters():
        #     print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}, {p.requires_grad}")
        # optimizer.step()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # print(output)
        # print(output.shape)
        metrics.update(output, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        num_updates += 1
        if steps_remaining > 0:
            for c, group in enumerate(optimizer.param_groups):
                group["lr"] += step_factors[c]
            steps_remaining = lr_reset_steps - 1
        else:
            # TODO: this will only work for timm schedulers
            lr_scheduler.step_update(num_updates=num_updates, metric=loss)

        # if argmax.std() == 0:
        #     log.error(f"ISSUE WITH NETWORK printing debugging info")
        #     for n, p in model.named_parameters():
        #         print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        #     raise ValueError
        argmax = torch.argmax(output, dim=1).to(torch.float32)
        # if argmax.std() == 0 and epoch > 5:
        #     if dist.get_rank() == 0:
        #         for n, p in model.named_parameters():
        #             print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        #     raise ValueError("Std == 0")

        if (i % config.training.print_freq == 0 or i == len(train_loader) - 1) and config["rank"] == 0:
            argmax = torch.argmax(output, dim=1).to(torch.float32)
            log.info(
                f"Argmax outputs s "
                f"mean: {argmax.mean().item():.5f}, max: {argmax.max().item():.5f}, "
                f"min: {argmax.min().item():.5f}, std: {argmax.std().item():.5f}",
            )
            progress.display(i + 1, log=log)

        optimizer.zero_grad()

    # if config.rank == 0:
    #     for n, p in model.named_parameters():
    #         if n.endswith("bias"):
    #             print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
    #     raise ValueError
    if config.rank == 0:
        log.info(f"Data Loading Time avg: {data_time.avg}")

    if config.rank == 0:
        columns = Columns(print_norms, equal=True, width=20)
        console = Console(width=100)
        console.print(columns)
    #     table = Table(title="Param grads")

    #     table.add_column("Name")
    #     table.add_column("param norm")

    #     dictionary={"Lionel Messi":35, "Cristiano Ronaldo":37, "Raheem Sterling":28, "Kylian Mbappé":24, "Mohamed Salah":30}

    #     for name,age in dictionary.items():
    #         table.add_row(name,str(age))

    #     console = Console()
    #     console.print(table)

    # if step_factors is not None and steps_remaining > 0:
    #     optimizer.opt1.param_groups[0]["lr"] = step_factors[2]
    #     if len(optimizer.opt1.param_groups) > 1:
    #         optimizer.opt1.param_groups[1]["lr"] = step_factors[2]
    #     optimizer.sigma_opt.param_groups[0]["lr"] = step_factors[3]

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if config["rank"] == 0 and config.enable_tracking:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        wandb_logger.log(
            {
                "train/loss": ls,
                "train/top1": t1,
                "train/top5": t5,
                "train/time": model_time,
            },
            # step=(epoch + 1) * len_train,
        )
        # mlflow.log_metrics(
        #     metrics={
        #         "train loss": ls,
        #         "train top1": t1,
        #         "train top5": t5,
        #         "train_time": model_time,
        #     },
        #     step=epoch,
        # )
    return losses.avg, loss, refactory_warmup


@torch.no_grad()
def save_selected_weights(network, epoch, config):
    if not config.baseline:
        raise RuntimeError(
            "Weights should only be saved when running baseline! (remove for other data gathering)",
        )
    if not config.save_weights:
        return
    save_list = config.save_layers
    save_location = Path(config.save_parent_folder) / config.model.name / "baseline_weights"
    rank = dist.get_rank()

    if rank != 0:
        dist.barrier()
        return
    # save location: resnet18/name/epoch/[u, s, vh, weights
    for n, p in network.named_parameters():
        if n in save_list:
            print(f"saving: {n}")
            n_save_loc = save_location / n / str(epoch)
            n_save_loc.mkdir(exist_ok=True, parents=True)

            torch.save(p.data, n_save_loc / "p.pt")
    # print("finished saving")
    dist.barrier()


@torch.no_grad()
def validate(val_loader, model, criterion, config, epoch, device, print_on_rank, pg, len_train, wandb_logger, metrics):
    def run_validate(loader, base_progress=0):
        # rank = 0 if not dist.is_initialized() else dist.get_rank()
        with torch.no_grad():
            end = time.time()
            num_elem = len(loader) - 1
            for i, data in enumerate(loader):
                if hasattr(config.data, "dali") and config.data.dali:
                    images = data[0]["data"]
                    target = data[0]["label"].squeeze(-1).long()
                else:
                    images = data[0]
                    target = data[1]
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                data_time.update(time.time() - end)

                if torch.any(torch.isnan(images)):
                    # for n, p in model.named_parameters():
                    # print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
                    raise ValueError("NaN in images... - VAL")

                # compute output
                output = model(images)
                loss = criterion(output, target)
                # argmax = torch.argmax(output.output, dim=1).to(torch.float32)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                metrics.update(output, target)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if torch.isnan(loss):
                    for n, p in model.named_parameters():
                        print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
                    raise ValueError("NaN loss - VAL")
                # argmax = torch.argmax(output, dim=1).to(torch.float32)
                # # print(
                # #     f"output mean: {argmax.mean().item()}, max: {argmax.max().item()}, ",
                # #     f"min: {argmax.min().item()}, std: {argmax.std().item()}",
                # # )
                # # progress.display(i + 1, log=log)

                # if argmax.std() == 0:
                #     log.error(f"ISSUE WITH NETWORK printing debugging info")
                #     for n, p in model.named_parameters():
                #         print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
                #     raise ValueError

                # if (i % config.training.print_freq == 0 or i == num_elem) and print_on_rank:
                if (i % 50 == 0 or i == num_elem) and print_on_rank:
                    argmax = torch.argmax(output, dim=1).to(torch.float32)
                    print(
                        f"output mean: {argmax.mean().item()}, max: {argmax.max().item()}, ",
                        f"min: {argmax.min().item()}, std: {argmax.std().item()}",
                    )
                    progress.display(i + 1, log=log)

                    # if argmax.std() == 0:
                    #     log.error(f"ISSUE WITH NETWORK printing debugging info")
                    #     for n, p in model.named_parameters():
                    #         print(f"{n}: {p.mean():.4f}, {p.min():.4f},
                    #         {p.max():.4f}, {p.std():.4f}")
                    #     raise ValueError

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE, pg=pg)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE, pg=pg)
    losses = AverageMeter("Loss", ":.4f", Summary.AVERAGE, pg=pg)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE, pg=pg)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE, pg=pg)
    progress = ProgressMeter(
        len(val_loader) + (len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset)),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()
    # if epoch == 5:
    #     for n, p in model.named_parameters():
    #         print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
    #     raise ValueError

    run_validate(val_loader)

    if len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * config["world_size"], len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=config.data["local_batch_size"],
            shuffle=False,
            num_workers=config["workers"],
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    progress.display_summary(log=log, printing_rank=print_on_rank)

    if config["rank"] == 0 and config.enable_tracking:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        wandb_logger.log(
            {
                "val/loss": ls,
                "val/top1": t1,
                "val/top5": t5,
            },
            # step=(epoch + 1) * len_train,
        )
        # mlflow.log_metrics(
        #     metrics={
        #         "val loss": ls,
        #         "val top1": t1,
        #         "val top5": t5,
        #     },
        #     step=epoch,
        # )
    if config.rank == 0:
        log.info(f"Data loading time avg: {data_time.avg}")

    return top1.avg, losses.avg


class LRWarmupLinear(object):
    def __init__(self, config, optimizer, model) -> None:
        self.min_lr = config.training.min_lr
        self.steps = config.training.lr_warmup_steps
        if config.training.lr_warmup_step_size is not None:
            self.add_value = config.training.lr_warmup_step_size
            self.max_lr = self.min_lr + (self.steps * self.add_value)
        else:
            self.max_lr = config.training.max_lr
            self.add_value = (self.max_lr - self.min_lr) / self.steps
        self.current_step = 0
        self.warming_up = True
        self.optimizer = optimizer
        self.config = config
        for group in self.optimizer.param_groups:
            group["lr_step"] = self.add_value
            group["lr_max"] = self.max_lr
            group["lr"] = self.min_lr

        # for last layer group -> need to adjust step as well (need a different max value)
        # self.adjust_last_layer_lr(model=model, config=config)

        self.sigma_steps = -1
        if not config.baseline:
            self.sigma_min = config.training.sigma_optimizer.min_lr
            self.sigma_steps = config.training.sigma_optimizer.steps
            self.optimizer.param_groups[-1]["lr"] = self.sigma_min

            if config.training.sigma_optimizer.lr_warmup_step_size is not None:
                self.sigma_add = config.training.sigma_optimizer.lr_warmup_step_size
                self.sigma_max = self.sigma_min + (self.sigma_steps * self.sigma_add)
            else:
                self.sigma_max = config.training.sigma_optimizer.max_lr
                self.sigma_add = (self.sigma_max - self.sigma_min) / self.sigma_steps
            sigma_group = self.optimizer.param_groups[-1]
            sigma_group["lr_step"] = self.sigma_add
            sigma_group["lr_max"] = self.sigma_max

    def adjust_last_layer_lr(self, model, config):
        # todo: make this better to actually do this programatically...
        if not config.baseline:
            raise RuntimeError("if not baseline, fix this function!!")
        param_list = [[n, p] for n, p in model.named_parameters()]
        last_name = param_list[-1][0].split(".")[:-1]
        last_name = ".".join(last_name)
        # get params of the last layer
        last_layer_params = []
        for n, p in param_list:
            if n.startswith(last_name) and p.requires_grad:
                last_layer_params.append(p)
        # TODO: update this to work with the other param groups from SVD...
        group0 = self.optimizer.param_groups[0]
        group0["params"] = group0["params"][: -1 * len(last_layer_params)]
        self.optimizer.add_param_group({"params": last_layer_params, "lr": self.min_lr})
        last_layer_group = self.optimizer.param_groups[-1]
        last_layer_group["lr_max"] = self.max_lr * 0.1
        last_layer_group["lr_step"] = self.add_value
        # print(self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[1]["lr"])

    @torch.no_grad()
    def step(self):
        self.current_step += 1
        # if self.current_step > self.steps:
        #     self.warming_up = False
        #     return
        grps = self.optimizer.param_groups
        if not self.config.baseline:
            grps = grps[:-1]
        if self.current_step <= self.steps:
            # if grps[0]["lr"] < self.max_lr:
            # TODO: end warmup and dont go here anymore??
            for group in grps:
                if group["lr"] >= group["lr_max"]:
                    group["lr"] = group["lr_max"]
                else:
                    group["lr"] += group["lr_step"]
        if not self.config.baseline and self.current_step <= self.sigma_steps:
            sigma_group = self.optimizer.param_groups[-1]
            sigma_group["lr"] += self.sigma_add
            if sigma_group["lr"] > self.sigma_max:
                sigma_group["lr"] = self.sigma_max


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE, pg=None):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()
        self.pg = pg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False, group=self.pg)
        self.sum, self.count = total.tolist()
        # self.avg = self.sum / self.count
        self.avg = total[0] / total[1]  # self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        # fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def display(self, batch, log):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # if self.rank == 0:
        #     # log.info("\t".join(entries))
        log.info(" ".join(entries))

    def display_summary(self, log, printing_rank):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if printing_rank:
            # print(" ".join(entries))
            # console.print(" ".join(entries))
            log.info(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
