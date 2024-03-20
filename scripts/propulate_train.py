import gc
import json
import logging
import os
import random
import select
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import copy

import numpy as np
import propulate
import torch.distributed as dist
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, errors, open_dict
from propulate.utils import get_default_propagator

import madonna
from madonna.utils import rgetattr, rsetattr


def objective(search_params, comm=MPI.COMM_WORLD):
    # load parameters from OS config file
    # rank = comm.Get_rank()
    conf_file = os.environ["CONFIG_NAME"]
    config = OmegaConf.load(conf_file)
    with open_dict(config):
        try:
            config.slurm_id = os.environ["SLURM_JOBID"]
        except KeyError:
            pass
        # overwrite the params from the search space into the config file
        for param in search_params:
            # print(f"{param} -> {rgetattr(config, param)} {search_params[param]}")
            rsetattr(config, param, search_params[param])
            # config[param] = search_params[param]
        # set name for the run to be all of what we expect
        config.name = "propulate-ab-test"
        # tags the runs to enable keeping propulate runs together
        tags = ["propulate"]
        if "tags" in config.tracking:
            tag = config.tracking.tags
            if isinstance(tag, list):
                tags.extend(tag)
            else:
                tags.append(tag)
        config.tracking.tags = tags
    ret_dict = madonna.trainers.ab_trainer.main(config, comm=comm)
    gc.collect()
    # print(ret_dict)
    return ret_dict["val_top1"]


def objective_subprocess(search_params, comm=MPI.COMM_WORLD):
    # print('start of obj', comm, comm.Iprobe())
    # ---------------- load base config + overwrite with params from propulate --------------------
    conf_file = os.environ["CONFIG_NAME"]
    config = OmegaConf.load(conf_file)
    comm.barrier()
    with open_dict(config):
        try:
            config.slurm_id = os.environ["SLURM_JOBID"]
        except KeyError:
            pass
        # overwrite the params from the search space into the config file
        for param in search_params:
            # print(f"{param} -> {rgetattr(config, param)} {search_params[param]}")
            rsetattr(config, param, search_params[param])
            # config[param] = search_params[param]
        # set name for the run to be all of what we expect
        config.name = "propulate-ab-test"
        # tags the runs to enable keeping propulate runs together
        tags = ["propulate"]
        if "tags" in config.tracking:
            tag = config.tracking.tags
            if isinstance(tag, list):
                tags.extend(tag)
            else:
                tags.append(tag)
        config.tracking.tags = tags
        # ---------------- END  load base config + overwrite with params from propulate ---------------
        #   set up new hostname/rank/size in config
        master_address = socket.gethostname()
        # print('before bcast')
        master_address = comm.bcast(str(master_address), root=0)
        # print('after bcast')

        # save env vars
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = "29500"
        # save the config vars
        config.worker_hostname = master_address
        config.worker_rank = comm.rank
        config.worker_size = comm.size

    # TODO: fix from below here!
    mpi_world = MPI.COMM_WORLD
    propulate_worker_rank = mpi_world.rank // comm.size  # assumes all the same size
    time.sleep(propulate_worker_rank / 100)
    # print(f"propulate_worker rank: {propulate_worker_rank}, {mpi_world.rank}, {comm.size}")
    os.environ["PROP_WORKER_RANK"] = str(propulate_worker_rank)
    os.environ["PROP_SUBWORKER_RANK"] = str(comm.rank)
    os.environ["PROP_DATASET"] = str(config.data.dataset)

    fileroot = Path(f"/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/configs/tmp/{config.data.dataset}")
    # print(fileroot)
    fileroot.mkdir(exist_ok=True)
    in_config = fileroot / f"propulate_start_{propulate_worker_rank}.yaml"
    kill_file = fileroot / f"{propulate_worker_rank}-kill-file.txt"
    with open(in_config, "w") as f:
        OmegaConf.save(config, f)

    output_file = fileroot / f"{propulate_worker_rank}-output.txt"
    comm.barrier()

    num_tries = 3
    ret_val = None
    kill = 0
    timeout = 2
    while num_tries > 0:
        try:
            os.remove(str(output_file))
        except FileNotFoundError:
            pass
        p = subprocess.Popen(
            ["python", "-u", "scripts/prop_train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=True,
            env=os.environ,
            shell=False,
        )
        kill_jobs = 0
        try:
            os.remove(str(kill_file))
        except FileNotFoundError:
            pass
        c = 0
        while not output_file.exists():
            # output = p.stdout.readline().decode()
            ready_out, _, _ = select.select([p.stdout], [], [], timeout)
            # print(f"after ready out: {ready_out}")
            out = ""
            while ready_out:
                nextline = p.stdout.readline().decode()
                if not nextline:
                    break
                if comm.rank == 0:
                    out += "o " + nextline
                    # print("o", out[:-1])
                ready_out, _, _ = select.select([p.stdout], [], [], timeout)
            if out:
                print(out)

            ready_errs, _, _ = select.select([p.stderr], [], [], timeout)
            # print(f"after ready err: {ready_errs}")
            errs = ""
            while ready_errs:
                nextline = p.stderr.readline().decode()
                if kill_jobs == 0:
                    for error_str in ["rror", "timed out", "ProcessGroupNCCL.cpp"]:
                        if error_str in nextline:
                            print("detected errors")
                            kill_jobs += 1
                if not nextline:
                    break
                errs += "e " + nextline
                ready_errs, _, _ = select.select([p.stderr], [], [], timeout)
            if errs:
                print(errs)
            if kill_jobs > 0:
                with open(kill_file, "w") as f:
                    f.write(str(kill_jobs))

            if output_file.exists() or kill_file.exists():
                # fine to have on all procs, file is on FS
                break

            time.sleep(2)
            c += 1
        # after the training, wait for everyone to join up
        comm.barrier()
        num_tries -= 1
        if output_file.exists():
            p.kill()
            try:
                with open(output_file, "r") as outputs:
                    ret_dict = json.loads(outputs.read())
                ret_val = ret_dict["val_top1"]
                num_tries = 0
                # successful run, exit loop
                break
            except json.decoder.JSONDecodeError:
                # nothing to do here execpt try again
                print(f"Issue loading json, removing file and trying again: {num_tries}")
                try:
                    os.remove(str(output_file))
                except FileNotFoundError:
                    pass
        if kill_file.exists():
            # if kill_jobs > 0:
            print(f"killing workers, errors. restarting, number of tries remaining: {num_tries}")
            p.kill()
            time.sleep(5)
    if ret_val is None:
        ret_val = 10

    return ret_val


def watch_file(filename, target_string):
    while True:
        with open(filename, "r") as file:
            contents = file.read()
            if target_string in contents:
                # print("String found!")
                break
        time.sleep(5)  # Adjust polling interval as needed


def stage_data(path="/hkfs/home/dataset/datasets/CIFAR100/"):
    path = Path(path)
    if rank % 4 != 0:
        return  # f"{os.environ['TMP']}/CIFAR10"
    dest = Path(f"{os.environ['TMP']}/cifar10/CIFAR10")
    print(f"Creating directory on tmp: {dest}")
    # test = Path("/hkfs/home/dataset/datasets/CIFAR10/train/cifar-10-batches-py/data_batch_1")
    # print(test.exists())
    dest.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    def copy_file(src_path, dest_dir):
        # copy source file to dest file
        _ = copy(src_path, dest_dir)

    with ThreadPoolExecutor(max_workers=70) as executor:
        futures = []
        for f in path.rglob("*"):
            if not f.is_file():
                continue
            futures.append(executor.submit(copy_file, f, dest))
            # print('staging', f)
        for fut in futures:
            _ = fut.result()
    print(f"Staging time: {time.perf_counter() - start_time}")


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 4)
    # stage_data()

    conf_file = os.environ["CONFIG_NAME"]
    config = OmegaConf.load(conf_file)

    # print(config.propulate.search_space)
    num_generations = 200000
    pop_size = 2 * MPI.COMM_WORLD.size
    # GPUS_PER_NODE = 4
    rng = random.Random(MPI.COMM_WORLD.rank)
    # propulate:
    #     ranks_per_worker: 4  # 1 node per worker, 4 workers/node
    #     num_islands: 2
    #     migration_probability: 0.9
    #     crossover: 0.7
    #     mutation: 0.4
    #     random: 0.1
    #     pollination: True
    # islands = MPI.COMM_WORLD.size // config.propulate.ranks_per_worker

    def uniquify(path):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + str(counter) + extension
            counter += 1
        return path

    log_path = Path(config.propulate.log_path)
    log_path.mkdir(exist_ok=True, parents=True)
    propulate.set_logger_config(
        level=logging.INFO,
        log_file=uniquify(log_path / "propulate.log"),
        log_to_stdout=True,
        colors=True,
    )
    propagator = get_default_propagator(
        pop_size,
        config.propulate.search_space,
        mate_prob=config.propulate.crossover,
        mut_prob=config.propulate.mutation,
        random_prob=config.propulate.random,
        rng=rng,
    )
    islands = propulate.Islands(
        loss_fn=objective_subprocess,  # Loss function to be minimized
        propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
        rng=rng,  # Separate random number generator for Propulate optimization
        generations=num_generations,  # Overall number of generations
        num_islands=config.propulate.num_islands,  # Number of islands
        migration_probability=config.propulate.migration_probability,  # Migration probability
        pollination=config.propulate.pollination,  # Whether to use pollination or migration
        checkpoint_path=Path(config.propulate.checkpoint_path),  # Checkpoint path
        # ----- SPECIFIC FOR MULTI-RANK UCS ----
        ranks_per_worker=config.propulate.ranks_per_worker,  # Number of ranks per (multi rank) worker
    )
    # TODO: set this up to change propulate's logging debug level!!!
    islands.evolve(top_n=1, logging_interval=1, debug=1)
