import logging
import os

import hydra
import torch
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.pretty import pprint

import madonna
import wandb
from madonna import utils

try:
    config_name = os.environ["CONFIG_NAME"]
except KeyError:
    config_name = "train.yaml"

# A logger for this file
log = logging.getLogger(__name__)


def main():
    conf_file = os.environ["CONFIG_NAME"]
    config = OmegaConf.load(conf_file)

    if "common" in config:
        # load the common dict
        common = OmegaConf.load(config.common)
        config = OmegaConf.merge(common, config)

    with open_dict(config):
        try:
            config.slurm_id = os.environ["SLURM_JOBID"]
        except KeyError:
            pass

    OmegaConf.set_struct(config, True)
    with open_dict(config):
        try:
            config.slurm_id = os.environ["SLURM_JOBID"]
        except KeyError:
            pass

    if config.training.trainer == "ortho_fix_train":
        fn = madonna.trainers.ortho_fix_train.main
    elif config.training.trainer == "patchwork_trainer":
        fn = madonna.trainers.patchwork_trainer.main
    elif config.training.trainer == "ab_trainer":
        fn = madonna.trainers.ab_trainer.main
    elif config.training.trainer == "fed_train":
        fn = madonna.trainers.fed_train.main
    else:
        raise ValueError(f"unknown trainer: {config.training.trainer}")

    fn(config)


if __name__ == "__main__":
    main()
