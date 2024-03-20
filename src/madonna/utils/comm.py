from __future__ import annotations

import datetime as dt
import os
import socket
import time

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT
from torch.distributed.distributed_c10d import (
    _new_process_group_helper,
    _pg_group_ranks,
    _store_based_barrier,
)

from . import utils

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_ROOT = 0


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size()
    else:
        size = 1
    return size


def get_world_rank():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 1
    return rank


def get_data_parallel_size():
    """
    Gets size of DP communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size(group=_DATA_PARALLEL_GROUP)
    else:
        size = 1
    return size


def get_data_parallel_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=_DATA_PARALLEL_GROUP)
    else:
        rank = 0
    return rank


def get_data_parallel_root(global_rank=False):
    if dist.is_available() and dist.is_initialized():
        if global_rank:
            root = _DATA_PARALLEL_ROOT
        else:
            root = 0
    else:
        root = 0
    return root


def get_local_rank():
    """
    Gets node local rank or returns zero if distributed is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0

    # number of GPUs per node
    if torch.cuda.is_available():
        local_rank = dist.get_rank(group=_DATA_PARALLEL_GROUP) % torch.cuda.device_count()
    else:
        local_rank = 0

    return local_rank


def get_data_parallel_group():
    if dist.is_available() and dist.is_initialized():
        grp = _DATA_PARALLEL_GROUP
    else:
        grp = None
    return grp


def get_local_size():
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    if torch.cuda.is_available():
        local_size = torch.cuda.device_count()
        # be sure to not return something bigger than world size
        local_size = min([local_size, get_world_size()])
    else:
        local_size = 1

    return local_size


def init_local_group(batchnorm_group_size, batchnorm_group_stride=1):
    # get comm stats
    my_rank = get_world_rank()
    world_size = get_world_size()

    # create local group
    num_groups = world_size // batchnorm_group_size
    assert (
        get_data_parallel_size() % batchnorm_group_size == 0
    ), "Error, make sure that the batchnorm group size is evenly divides the data parallel size"
    assert (
        get_data_parallel_size() >= batchnorm_group_size
    ), "Error, make sure the batchnorm groups do not extend beyond data parallel groups"
    local_group = None
    if world_size > 1 and batchnorm_group_size > 1:
        num_stride_groups = num_groups // batchnorm_group_stride
        local_groups = []
        for i in range(num_stride_groups):
            for j in range(batchnorm_group_stride):
                start = j + i * (batchnorm_group_size * batchnorm_group_stride)
                end = start + batchnorm_group_size * batchnorm_group_stride
                ranks = list(range(start, end, batchnorm_group_stride))
                local_groups.append(ranks)
                tmp_group = dist.new_group(ranks=ranks)
                if my_rank in ranks:
                    local_group = tmp_group

    return local_group


# split comms using MPI
def init_split(
    method,
    instance_size,
    split_groups=True,
    batchnorm_group_size=1,
    batchnorm_group_stride=1,
    verbose=False,
    directory=None,
):
    # import MPI here:
    from mpi4py import MPI

    # data parallel group
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT

    # get MPI stuff
    mpi_comm = MPI.COMM_WORLD.Dup()
    comm_size = mpi_comm.Get_size()
    comm_rank = mpi_comm.Get_rank()

    # determine the number of instances
    num_instances = comm_size // instance_size
    # determine color dependent on instance id:
    # comm_rank = instance_rank +  instance_id * instance_size
    instance_id = comm_rank // instance_size
    instance_rank = comm_rank % instance_size

    # split the communicator
    mpi_instance_comm = mpi_comm.Split(color=instance_id, key=instance_rank)

    # for a successful scaffolding, we need to retrieve the IP addresses
    port = 29500
    master_address = socket.gethostname()
    if split_groups:
        master_address = mpi_instance_comm.bcast(master_address, root=0)
    else:
        master_address = mpi_comm.bcast(master_address, root=0)

    # save env vars
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = str(port)

    # special stuff for file wireup method
    if method == "nccl-file":
        master_filename = os.path.join(directory, f"instance{instance_id}.store")
        if comm_rank == 0:
            os.makedirs(directory, exist_ok=True)
        mpi_comm.Barrier()

        # delete the wireup file if it exists
        if (instance_rank == 0) and os.path.isfile(master_filename):
            os.remove(master_filename)
        mpi_instance_comm.Barrier()

    # set the parameters depending on whether we want to split or not
    if split_groups:
        nccl_world_size = instance_size
        nccl_world_rank = instance_rank
    else:
        nccl_world_size = comm_size
        nccl_world_rank = comm_rank

    # do the dist init (if we have non trivial instances)
    if instance_size > 1:
        if verbose and instance_rank == 0:
            print(
                f"Starting NCCL wireup for instance {instance_id} with method {method}",
                flush=True,
            )
        # dangerous but necessary: done in run.sub now
        # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        if method == "nccl-slurm":
            # get TCP Store
            wireup_store = dist.TCPStore(
                host_name=master_address,
                port=port,
                world_size=nccl_world_size,
                is_master=(nccl_world_rank == 0),
                timeout=dt.timedelta(seconds=3600),
            )
        else:
            raise NotImplementedError(
                f"Error, unknown wireup method {method}, supported are [nccl-slurm, nccl-file]",
            )

        # initialize group
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=nccl_world_size,
            rank=nccl_world_rank,
        )

        if split_groups:
            _DATA_PARALLEL_GROUP = None
            _DATA_PARALLEL_ROOT = 0
        else:
            # create data parallel group:
            for inst_id in range(num_instances):
                start = inst_id * instance_size
                end = start + instance_size
                ranks = list(range(start, end))
                tmp_group = dist.new_group(ranks=ranks)
                if inst_id == instance_id:
                    _DATA_PARALLEL_GROUP = tmp_group
                    _DATA_PARALLEL_ROOT = ranks[0]

        # make sure to call a barrier here in order for sharp to use the default comm:
        dist.barrier(device_ids=[get_local_rank()], group=_DATA_PARALLEL_GROUP)
        # the nccl wireup call could be non blocking, so we wait for the first barrier
        # to complete before printing this message
        if verbose and instance_rank == 0:
            print(f"Completed NCCL wireup for instance {instance_id}", flush=True)

    # get the local process group for batchnorm
    batchnorm_group = init_local_group(batchnorm_group_size, batchnorm_group_stride)

    return mpi_comm, mpi_instance_comm, instance_id, batchnorm_group


def split_process_group_node_local(num_gpus_per_node=4):
    """
    Splits a PyTorch process group into node-local groups with ranks 0..(num_gpus_per_node - 1) on each node
    and a group which holds rank N on all nodes (i.e. all local-rank 0 process in group0, all local-rank 1, etc.)

    Args:
        num_gpus_per_node (int, optional): Number of GPUs per node. Defaults to 4.

    Returns:
        dict: {
            "local": node local group,
            f"only{N}": group with only the processes with local rank N -> if no group there (not member) == False
        }
    """
    global_rank = dist.get_global_rank()
    world_size = dist.get_world_size()

    # 1. Determine the node information
    num_nodes = world_size // num_gpus_per_node
    node_id = global_rank // num_gpus_per_node

    # 2. Create node-local process groups
    node_ranks = [node_id * num_gpus_per_node + gpu_id for gpu_id in range(num_gpus_per_node)]
    node_local_group = dist.new_group(ranks=node_ranks)

    groups = {"local": node_local_group}

    # 3. Determine the rank within the node-local group
    for g in range(num_gpus_per_node):
        groups[f"only{g}"] = dist.new_group(ranks=[g * n for n in num_nodes])
        if groups[f"only{g}"] is None:
            groups[f"only{g}"] = False

    return groups


# do regular init
def init(
    method,
    ranks_per_gpu=1,
    batchnorm_group_size=1,
    batchnorm_group_stride=1,
    mpi_comm=None,
    hostname=None,
    rank=None,
    size=None,
):
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT
    port = 29500
    # print('h', mpi_comm, hostname)
    if hostname is not None:
        comm_size = size
        comm_rank = rank
        master_address = hostname
    else:
        if mpi_comm is None:
            from mpi4py import MPI

            mpi_comm = MPI.COMM_WORLD

        # get master address and port
        # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

        comm_size = mpi_comm.Get_size()
        master_address = socket.gethostname()
        master_address = mpi_comm.bcast(str(master_address), root=0)

        # save env vars
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = str(port)

        comm_rank = mpi_comm.Get_rank()

    nccl_world_size = comm_size
    nccl_world_rank = comm_rank
    # print(mpi_comm.rank, mpi_comm.size, master_address, port)
    # exit(0)

    if method == "nccl-openmpi":
        # addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        # use that URI
        # address = addrport.split(":")[0]
        # use the default pytorch port
        # os.environ["MASTER_ADDR"] = address
        rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 0))

        # init DDP
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

    elif method == "nccl-slurm":
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"device count: {torch.cuda.device_count()}, device number: {comm_rank % 4}")
        torch.cuda.set_device(comm_rank % 4)
        time.sleep(0.01 * comm_rank)

        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=nccl_world_size,
            is_master=(nccl_world_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=nccl_world_size,
            rank=nccl_world_rank,
        )
    elif method == "gloo":
        time.sleep(0.001 * comm_rank)

        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=nccl_world_size,
            is_master=(nccl_world_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="gloo",
            store=wireup_store,
            world_size=nccl_world_size,
            rank=nccl_world_rank,
        )
    else:
        raise NotImplementedError()

    # make sure to call a barrier here in order for sharp to use the default comm:
    if dist.is_initialized():
        if ranks_per_gpu > 1 and method != "gloo":
            torch.cuda.set_device(get_local_rank() // ranks_per_gpu)
        elif method == "gloo":
            pass
        else:
            torch.cuda.set_device(get_local_rank())

        dist.barrier()
        disttest = torch.ones(1)
        if method != "gloo":
            disttest = disttest.cuda()

        dist.all_reduce(disttest)
        assert disttest[0] == nccl_world_size, "failed test of dist!"
    else:
        disttest = None

    # get the local process group for batchnorm
    batchnorm_group = init_local_group(batchnorm_group_size, batchnorm_group_stride)

    print(f"finished dist init - rank: {dist.get_rank()} ws: {dist.get_world_size()}, test: {disttest}")
    return batchnorm_group


def create_sub_groups(group_size: int) -> dist.ProcessGroup:
    """
    Create local sub-groups in the communicator.
    NOTE: only the local group will be returned on each process, all procs will be in a local group

    Parameters
    ----------
    group_size: int
        size of groups to create

    Returns
    -------
    torch.distributed.ProcessGroup
    """
    from mpi4py import MPI

    global_size = dist.get_world_size()
    global_rank = dist.get_rank()

    assert global_size % group_size == 0, f"global_size % group_size != 0 ({global_size}, {group_size})"

    global _pg_group_ranks

    group_id = global_rank // group_size
    group_rank = global_rank % group_size
    time.sleep(global_rank * 0.01)

    mpi_comm = MPI.COMM_WORLD
    gp_ranks = [i for i in range(group_id * group_size, (group_id + 1) * group_size)]
    # my_groups_rank0 =

    group = mpi_comm.group.Incl(gp_ranks)
    mpi_group = mpi_comm.Create_group(group)
    master_address = socket.gethostname()
    # if mpi_group.Get_rank() != 0:
    # master_address = None
    master_address = mpi_group.bcast(master_address, root=0)
    # print(master_address)

    # save env vars
    os.environ["MASTER_ADDR"] = master_address
    port = 29510 + group_id
    os.environ["MASTER_PORT"] = str(port)
    # print(master_address, port)

    ranks = torch.arange(global_size).tolist()
    grp_st, grp_sp = group_id * group_size, (group_id + 1) * group_size
    local_ranks = ranks[grp_st:grp_sp]

    # ------- from torch.distributed --------------------------------
    wireup_store = dist.TCPStore(
        host_name=master_address,
        port=port,
        world_size=group_size,
        is_master=(group_rank == 0),
        timeout=dt.timedelta(seconds=3600),
    )
    # pg = dist.new_group(ranks=local_ranks)
    pg = _new_process_group_helper(
        group_size,
        group_rank,
        local_ranks,
        backend="gloo",
        store=wireup_store,
        pg_options=None,
        timeout=_DEFAULT_PG_TIMEOUT,
    )

    # Create the global rank to group rank mapping
    _pg_group_ranks[pg] = {global_rank: group_rank for group_rank, global_rank in enumerate(local_ranks)}
    pg._set_sequence_number_for_group()
    return pg


def init_and_set_config_rank_size(config, mpi_comm=None):
    # if mpi_comm is None:
    #     from mpi4py import MPI
    #     mpi_comm = MPI.COMM_WORLD
    # elif isinstance(mpi_comm, str):
    #     mpi_comm = None

    size = 1
    rank = 0
    hostname = None
    if "worker_hostname" in config:
        hostname = config.worker_hostname
        rank = int(config.worker_rank)
        size = int(config.worker_size)
    print(f"in comm: {hostname} {rank} {size}")
    # init will return a batchnorm group if we want, but we dont really care with this function
    #   if you want one, make adjustments here
    if "comm_method" in config and config.comm_method == "gloo":
        init(method="gloo", mpi_comm=mpi_comm, hostname=hostname, rank=rank, size=size)
        rank = dist.get_rank()
        size = dist.get_world_size()
        return rank, size
    try:
        if int(os.environ["SLURM_NTASKS"]) > 1 or int(os.environ["OMPI_COMM_WORLD_SIZE"]) > 1:
            init(method="nccl-slurm", mpi_comm=mpi_comm, hostname=hostname, rank=rank, size=size)
            rank = dist.get_rank()
            size = dist.get_world_size()
    except KeyError:
        try:
            if int(os.environ["OMPI_COMM_WORLD_SIZE"]) > 1:
                init(method="nccl-slurm", mpi_comm=mpi_comm, hostname=hostname, rank=rank, size=size)
                rank = dist.get_rank()
                size = dist.get_world_size()
        except KeyError as e:
            raise e

    return rank, size


def average_spec_objects_in_model(model, object_names: list, group=dist.group.WORLD):
    """Averages parameters of specified layers across multiple workers in a PyTorch distributed setup.

    Args:
        model (torch.nn.Module): The PyTorch model containing the layers.
        object_names (list): List of names of the model objects to average.
        group (dist.ProcessGroup, optional): The process group for communication. Defaults to WORLD.
    """
    waits = []
    for name in object_names:
        # Ensure the selected layer has parameters to average
        # if hasattr(model.modules(), f"module_{idx}") and list(model.module_[idx].parameters()):

        params = list(utils.rgetattr(model, name).parameters())

        # Sum the parameters across all workers
        for p in params:
            p.data /= dist.get_world_size(group)
            waits.append(dist.all_reduce(p.data, op=dist.ReduceOp.SUM, group=group, async_op=True))

    # Average the parameters in place
    for w in waits:
        w.wait()
