#!/usr/bin/env bash

# Slurm job configuration
####SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
####SBATCH --job-name=madonna-image-ab
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-madonna
#SBATCH --output="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/logs/slurm/slurm-%j"
#### SBATCH --exclude=hkn[0431,0702,0704]

# export CONFIG_NAME="configs/cifar10/vgg16/baseline_2nodes.yaml"
# export CONFIG_NAME="configs/cifar10/vgg16/ab_2nodes.yaml"

# export CONFIG_NAME="configs/imagenet/vit/ab_4nodes_bs4k.yaml"
# export CONFIG_NAME="configs/imagenet/vit/baseline_bs32k.yaml"

# export CONFIG_NAME="configs/imagenet/resnet/ab_32nodes_bs32k0.01.yaml"
# export CONFIG_NAME="configs/imagenet/resnet/baseline_bs32k0.01.yaml"

# export CONFIG_NAME="configs/ab_train_prop_im_rs.yaml"
# export TRAIN_SCRIPT="scripts/propulate_train.py"

export TRAIN_SCRIPT="scripts/singularity_train.py"


ml purge

BASE_DIR="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/"

TOMOUNT='/etc/slurm/task_prolog.hk:/etc/slurm/task_prolog.hk,'
TOMOUNT+="${BASE_DIR},"
TOMOUNT+="/hkfs/home/dataset/datasets/,"
export TOMOUNT="${TOMOUNT}"

SRUN_PARAMS=(
  --mpi="pmix"
  --gpus-per-task=1
  # --cpus-per-task=8
  #--cpu-bind="ldoms"
  --gpu-bind="closest"
  --label
)

SCRIPT_DIR="/pfs/work7/workspace/scratch/qv2382-madonna-ddp/qv2382-madonna-ddp/madonna"

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_API_KEY="4a4a69b3f101858c816995e6dfa553718fdf0dbe"
export MADONNA="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna"
export VENV_NAME="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/containers/test_env"
export SINGULARITY_FILE="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/containers/torch_new.sif"



# PATH=$PATH:/home/kit/scc/qv2382/.local/bin
# echo "--bind ${TOMOUNT} ${SINGULARITY_FILE} /usr/bin/sh -c export CUDA_VISIBLE_DEVICES=0,1,2,3; source ${VENV_NAME}/bin/activate; cd ${MADONNA}; CONFIG_NAME=${CONFIG_NAME} python scripts/singularity_train.py"
srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${TOMOUNT}" \
  "${SINGULARITY_FILE}" \
  bash -c "export CUDA_VISIBLE_DEVICES=0,1,2,3; source ${VENV_NAME}/bin/activate; cd ${MADONNA}; CONFIG_NAME=${CONFIG_NAME} python -u ${TRAIN_SCRIPT}"
