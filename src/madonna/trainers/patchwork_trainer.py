from __future__ import annotations

import logging
import os
import random
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from enum import Enum
from pathlib import Path

import hydra

# from mpi4py import MPI
import numpy as np

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
from omegaconf import OmegaConf, open_dict
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.pretty import pprint
from torch.autograd.grad_mode import no_grad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torchmetrics import MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall

import madonna
import wandb

from .lr_dist_trainer import LowRankSyncTrainer
from .patchwork import PatchworkSVDTrainer

best_acc1 = 0
log = logging.getLogger(__name__)


def main(config):  # noqa: C901
    # log.info(config)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    print(hydra_cfg["runtime"]["output_dir"])
    # log.info("here is some log testing")
    # log.info("more stuff")
    # log.info("do i need more things?")
    # raise ValueError

    # =========================================== logging / tracking init ========================
    for handler in log.parent.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
    with open_dict(config):
        config["log_file_out"] = log_file
        if config.tracking.logging_rank == "all":  #
            config.tracking.logging_rank = config.rank

    wandb_run = False
    if config.tracker == "wandb":
        wandb_run = madonna.utils.tracking.check_wandb_resume(config)
    if config.training.resume and not wandb_run:  # resume an interrupted run
        ckpt = config.training.checkpoint
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"

    if config.training.checkpoint_out_root is not None:
        with omegaconf.open_dict(config):
            config.save_dir = madonna.utils.utils.increment_path(
                Path(config.training.checkpoint_out_root) / config.name,
            )  # increment run
        if config.rank == config.tracking.logging_rank:
            log.info(f"save_dir: {config.save_dir}")

        save_dir = Path(config.save_dir)

        # Directories
        ex = "" if not config.training.federated else str(dist.get_rank())
        wdir = save_dir / f"weights{ex}"
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        log.info(f"out_dir (wdir): {wdir}")
        last = wdir / "last.pt"
    else:
        wdir = None
    # results_file = save_dir / 'results.txt'
    # f = open(results_file, 'w')
    if config.rank == config.tracking.tracking_rank and config.enable_tracking:
        wandb_logger = madonna.utils.tracking.WandbLogger(config)
    else:
        wandb_logger = None

    # =========================================== END logging / tracking init ========================

    if dist.is_initialized():
        gpu = dist.get_rank() % torch.cuda.device_count()  # only 4 gpus/node
        log.debug(f"Using GPU: {gpu}")
    else:
        log.info(f"available GPUS: {torch.cuda.device_count()}")
        gpu = 0
        # log.info(f"available GPUS: {torch.cuda.device_count()}")
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    # =========================================== Seed setting =======================================
    if config.seed is None:
        seed = random.randint(0, 4294967295)
        # seed = torch.seed()
        tseed = torch.tensor(seed, dtype=torch.int64, device=device)
    else:
        seed = int(config.seed)
        tseed = torch.tensor(seed, dtype=torch.int64, device=device)

    if config.training.init_method == "random":
        seed += config.rank
    elif dist.is_initialized() and config.training.init_method in [
        "unified",
        "rand-sigma",
        "ortho-sigma",
        "sloped-sigma",
    ]:
        dist.broadcast(tseed, src=0)
    else:
        raise ValueError(
            f"config.training.init_method should be one of [random, unified, rand-sigma, ortho-sigma],"
            f" given: {config.training.init_method}",
        )
    seed = tseed.item()
    # TODO: should deterministic be True??
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.set_printoptions(precision=5)
    random.seed(seed)
    torch.manual_seed(seed)

    log.info(f"Seed: {seed}, init method: {config.training.init_method}")
    # if config.training.init_method == "random-simga":
    #     with open_dict(config):
    #         config.training.seed = seed
    # =========================================== END Seed setting =======================================

    # print('before get model')
    model = madonna.utils.get_model(config)
    if not config.cpu_training:
        model.cuda(gpu)

    if config.training.init_method.endswith("-sigma"):
        madonna.utils.utils.modify_model_random_simgas(
            model,
            device=gpu,
            mode=config.training.init_method,
        )

    # if config.training.sync_batchnorm and dist.is_initialized():
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # if config.training.federated and config.baseline:
    # log.info("Using federated data scheme, model is not DDP")
    # elif config.baseline:
    #     model = DDP(model)  # , device_ids=[config.rank])
    #     log.info("using DDP baseline model")
    # else:
    #     model_hold = hydra.utils.instantiate(config.training.fixing_method)
    #     model = model_hold(model).to(device)
    #     log.info("using SVD model")

    # model_param_dict = madonna.lrsync.sync.get_param_dict_with_svds(model)

    criterion = madonna.utils.get_criterion(config)
    optimizer = madonna.utils.get_optimizer(config, model, lr=config.training.lr)

    dset_dict = madonna.utils.datasets.get_dataset(config)
    train_loader, train_sampler = dset_dict["train"]["loader"], dset_dict["train"]["sampler"]
    val_loader = dset_dict["val"]["loader"]

    train_len = config.training.iterations_per_train
    scheduler, _ = madonna.utils.get_lr_schedules(config, optimizer, train_len)

    train_metrics = WrappedMetrics(epoch=0, len_loader=config.training.iterations_per_train)

    trainer = PatchworkSVDTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        lr_scheduler=scheduler,
        config=config,
        metrics=train_metrics,
    )
    if config.rank == 0:
        print(f"len dataloader: {len(trainer.infloader)}")

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
            # if not config.baseline:
            #     optimizer.param_groups[-1]["lr"] = config.training.sigma_optimizer.min_lr
            # # optimizer should have the correct LR from the loading point
            # if not config.baseline:
            #     if "next_stability_iteration" in checkpoint:
            #         model.next_stability_iteration = checkpoint["next_stability_iteration"]
            #     if "call_count" in checkpoint:
            #         model.call_count = checkpoint["call_count"]
            if scheduler is not None and start_epoch > 0:
                if True:  # args.sched_on_updates: FIXME
                    scheduler.step_update(start_epoch * len(train_loader))
                else:
                    scheduler.step(start_epoch)
        else:
            print(f"=> no checkpoint found at: {config.training.checkpoint}")

    # if config['evaluate']:
    #     validate(val_loader, dlrt_trainer, config)
    #     return

    val_metrics = MetricCollection(
        [
            MulticlassF1Score(num_classes=config.data.classes),
            MulticlassPrecision(num_classes=config.data.classes),
            MulticlassRecall(num_classes=config.data.classes),
        ],
    ).to(device)

    rank = dist.get_rank() if dist.is_initialized() else 0

    best_fitness = 0.0
    last_val_top1s = []

    for epoch in range(start_epoch, config.training["epochs"]):
        torch.cuda.reset_peak_memory_stats()
        if config["rank"] == config.tracking.tracking_rank:
            lr_list, prnt_str = get_lrs(optimizer)
            log.info(f"Begin epoch {epoch} LRs: {prnt_str}")
            lrs = {"lr": lr_list[0]}
            if config.enable_tracking:
                wandb_logger.log(lrs)
        if dist.is_initialized() and config.data.distributed_sample and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        trainer.metrics.reset(epoch, trainer.iterations_per_train)
        train_metrics, train_time = trainer.train()

        ls = train_metrics["loss"].item() if isinstance(train_metrics["loss"], torch.Tensor) else train_metrics["loss"]
        t1 = train_metrics["top1"].item() if isinstance(train_metrics["top1"], torch.Tensor) else train_metrics["top1"]
        t5 = train_metrics["top5"].item() if isinstance(train_metrics["top5"], torch.Tensor) else train_metrics["top5"]
        if rank == config.tracking.tracking_rank and config.enable_tracking:
            wandb_logger.log(
                {
                    "train/loss": ls,
                    "train/top1": t1,
                    "train/top5": t5,
                    "train/time": train_time,
                },
            )
        if rank == config.tracking.logging_rank:
            log.info(f"End Train params avg across procs: loss: {ls:.4f}\ttop1: {t1:.4f}\ttop5: {t5:.4f}")
        # evaluate on validation set
        val_top1, val_loss = validate(
            val_loader=val_loader,
            model=trainer.model_to_run,
            criterion=criterion,
            config=config,
            device=device,
            wandb_logger=wandb_logger if config.enable_tracking else None,
            metrics=val_metrics,
        )

        # if config.rank == 0:
        #     log.info(
        #         f"Average val loss across process space: {val_loss} " f"-> diff: {train_loss - val_loss}",
        #     )

        # log metrics
        val_metrics_epoch = val_metrics.compute()
        log_dict = {
            # "train/loss": train_loss,
            "val/f1": val_metrics_epoch["MulticlassF1Score"],
            "val/precision": val_metrics_epoch["MulticlassPrecision"],
            "val/recall": val_metrics_epoch["MulticlassRecall"],
            # "val/loss": val_loss,
        }

        # Save model
        if dist.is_initialized():
            wait = dist.barrier(async_op=True)

        if rank == config.tracking.logging_rank:
            log.info(
                f"Epoch end metrics: "
                # f"train f1/prec/rec: {log_dict['train/f1']:.4f} / "
                # f"{log_dict['train/precision']:.4f} / {log_dict['train/recall']:.4f}"
                f"val f1/prec/rec: {log_dict['val/f1']:.4f} / "
                f"{log_dict['val/precision']:.4f} / {log_dict['val/recall']:.4f}",
            )

        ckpt = {
            "epoch": epoch,
            "best_fitness": best_fitness,
            # 'training_results': results_file.read_text(),
            "model": model.state_dict(),
            # "optimizer": optimizer.state_dict(),  TODO: FIXME!!!!
        }

        if rank == config.tracking.tracking_rank and config.enable_tracking:
            last_val_top1s.append(val_top1.item() if isinstance(val_top1, torch.Tensor) else val_top1)
            if len(last_val_top1s) > 10:
                slope, _ = np.polyfit(x=np.arange(10), y=np.array(last_val_top1s[-10:]), deg=1)
                log.info(f"Slope of Top1 for last 10 epochs: {slope:.5f}")

            if val_top1 > best_fitness:
                best_fitness = val_top1

            wandb_logger.log(log_dict)
            wandb_logger.end_epoch(best_result=best_fitness == val_top1)

            ckpt["wandb_id"] = (wandb_logger.wandb_run.id if wandb_logger.wandb else None,)
            # Save last, best and delete
        if wdir is not None and config.training.save_checkpoints:  # TODO: saving from ALL ranks here
            torch.save(ckpt, last)
            # if best_fitness == val_top1:
            #     torch.save(ckpt, best)
            #     print("After 1st save")
            if best_fitness == val_top1:
                torch.save(ckpt, wdir / "best_{:03d}.pt".format(epoch))
                print("After best save")

            if epoch == 0:  # first
                torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                print("After 1st save")
            elif config.training.save_period != -1 and ((epoch + 1) % config.training.save_period) == 0:  # on command
                torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                print("After periodic save")
            # elif epoch >= (config.training.epochs - 5):
            #     torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))

            del ckpt
        if dist.is_initialized():
            # wait here for the saving and such...it didnt work to have it afterwards
            wait.wait(timeout=timedelta(seconds=60))
        # # early stopping for imagenet...
        # if (val_top1 < 15. and epoch >= 5) or \
        #    (val_top1 < 60. and epoch >= 25) or \
        #    (val_top1 < 70. and epoch >= 50):  # or \
        #     #    (val_top1 < 75. and epoch >= 70):  # or \
        #     # (val_top1 < 78. and epoch >= 100):
        #     if rank == 0:
        #         log.info("Early stopping")
        #     break
        scheduler.step(epoch + 1, metric=val_loss)
        val_metrics.reset()
    if rank == config.tracking.tracking_rank and config.enable_tracking:
        log.info("End of run")
        wandb_logger.finish_run()
    # import json
    # val_top1 = val_top1 if not isinstance(val_top1, torch.Tensor) else val_top1.item()
    # ret_dict = {"train_loss": train_loss, "train_top1": train_t1, "val_loss": val_loss, "val_top1": val_top1}
    # # propulate minimizes...
    # ret_dict["train_top1"] = 1 - (ret_dict["train_top1"] * 0.01)
    # ret_dict["val_top1"] = 1 - (ret_dict["val_top1"] * 0.01)
    # print("from train", ret_dict)
    # # out_file_root = Path("/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/configs/tmp/")
    # out_file = Path("/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/configs/tmp/")
    # with open(out_file / f"{os.environ['RANK']}-output.txt", "w") as convert_file:
    #     # convert_file.write(json.dumps(ret_dict))
    #     json.dump(ret_dict, convert_file)
    # return ret_dict


def get_lrs(opt):
    out_lrs = []
    prnt_str = ""
    for group in opt.param_groups:
        out_lrs.append(group["lr"])
        prnt_str += f"group {len(out_lrs)}: lr {group['lr']:.6f}\t"
    return out_lrs, prnt_str


@torch.no_grad()
def validate(val_loader, model, criterion, config, device, wandb_logger, metrics):
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
                if config.baseline:
                    output = model(images)
                else:
                    output, _ = model(images)
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

                # if (i % config.training.print_freq == 0 or i == num_elem) and print_on_rank:
                if (i % 50 == 0 or i == num_elem) and config.tracking.logging_rank == config.rank:
                    progress.display(i + 1, log=log)
                if i % 50 == 0 or i == num_elem:
                    argmax = torch.argmax(output, dim=1).to(torch.float32)
                    print(
                        f"output mean: {argmax.mean():.4f}, max: {argmax.max():.4f}, ",
                        f"min: {argmax.min():.4f}, std: {argmax.std():.4f}",
                    )

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4f", Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset)),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()
    vt1 = time.perf_counter()
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
    val_time_total = time.perf_counter() - vt1

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    progress.display_summary(log=log, printing_rank=config.tracking.logging_rank)

    if config["rank"] == config.tracking.tracking_rank and config.enable_tracking:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        wandb_logger.log(
            {
                "val/loss": ls,
                "val/top1": t1,
                "val/top5": t5,
                "val/total_time": val_time_total,
            },
        )
    if config.rank == config.tracking.logging_rank:
        log.info(f"Data loading time avg: {data_time.avg}")
        log.info(f"End Val avgs: Loss: {losses.avg:.4f} Top1: {top1.avg:.4f} Top5: {top5.avg:.4f}")

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class WrappedMetrics:
    def __init__(self, epoch, len_loader) -> None:
        self.reset(epoch, len_loader)

    def reset(self, epoch, len_loader):
        self.batch_time = AverageMeter("Time", ":6.3f")
        self.data_time = AverageMeter("Data", ":6.3f")
        self.losses = AverageMeter("Loss", ":.4f")
        self.top1 = AverageMeter("Acc@1", ":6.2f")
        self.top5 = AverageMeter("Acc@5", ":6.2f")
        self.progress = ProgressMeter(
            len_loader,
            [self.batch_time, self.data_time, self.losses, self.top1, self.top5],
            prefix=f"Epoch: [{epoch}]",
        )

    def __call__(self, outputs, labels, loss, sz):
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        self.losses.update(loss.item(), sz)
        try:
            self.top1.update(acc1[0], sz)
            self.top5.update(acc5[0], sz)
        except IndexError:
            self.top1.update(acc1, sz)
            self.top5.update(acc5, sz)

    def display(self, i, output=None):
        if output is not None:
            argmax = torch.argmax(output, dim=1).to(torch.float32)
            log.info(
                f"Argmax outputs s "
                f"mean: {argmax.mean().item():.5f}, max: {argmax.max().item():.5f}, "
                f"min: {argmax.min().item():.5f}, std: {argmax.std().item():.5f}",
            )
        self.progress.display(i + 1, log=log)

    def compute(self):
        self.losses.all_reduce()
        self.top1.all_reduce()
        self.top5.all_reduce()
        return {"loss": self.losses.avg, "top1": self.top1.avg, "top5": self.top5.avg}


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


def accuracy(output, target, topk=(1,), mixup=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        if not mixup:
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        else:
            maxk = max(topk)
            batch_size = target.size(0)
            if target.ndim == 2:
                target = target.max(dim=1)[1]

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[None])

            res = []
            for k in topk:
                correct_k = correct[:k].flatten().sum(dtype=torch.float32)
                res.append(correct_k * (100.0 / batch_size))
            return res
