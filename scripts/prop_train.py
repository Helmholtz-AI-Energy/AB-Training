def subproc_train():
    import json
    import os
    import time
    from pathlib import Path

    import torch
    from omegaconf import DictConfig, OmegaConf, errors, open_dict

    import madonna

    propulate_worker_number = os.environ["PROP_WORKER_RANK"]
    dataset = os.environ["PROP_DATASET"]
    fileroot = Path(f"/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/configs/tmp/{dataset}")
    in_config = fileroot / f"propulate_start_{propulate_worker_number}.yaml"
    # print(f"in_conf: {in_config}")
    config = OmegaConf.load(in_config)
    # print(config)
    with open_dict(config):
        config.rank = int(os.environ["PROP_SUBWORKER_RANK"])
        config.worker_rank = int(os.environ["PROP_SUBWORKER_RANK"])
    #  = str(comm.rank)
    ret_dict = madonna.trainers.ab_trainer.main(config, comm=None, subproc=True)
    for k in ret_dict:
        if isinstance(ret_dict[k], torch.Tensor):
            ret_dict[k] = ret_dict[k].item()
    # ret_dict = {'a': 1}

    print(f"{config.worker_rank} - output to {fileroot / propulate_worker_number}-output.txt")
    if config.worker_rank == 0:
        with open(fileroot / f"{propulate_worker_number}-output.txt", "w") as convert_file:
            # convert_file.write(json.dumps(ret_dict))
            json.dump(ret_dict, convert_file)
    exit(0)


if __name__ == "__main__":
    subproc_train()
