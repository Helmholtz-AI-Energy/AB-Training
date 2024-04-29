#!/usr/bin/env bash

# need to set the config name, job name, and node count for each job
# ViT strong scaling
# # CONFIG_NAME="configs/imagenet/vit/ab_2nodes_bs2k.yaml" sbatch -N 2 --job-name=madonna-image-ab-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
# CONFIG_NAME="configs/imagenet/vit/ab_4nodes_bs4k.yaml" sbatch  -N 4 --job-name=madonna-image-ab-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
# CONFIG_NAME="configs/imagenet/vit/ab_8nodes_bs8k.yaml" sbatch  -N 8 --job-name=madonna-image-ab-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
# CONFIG_NAME="configs/imagenet/vit/ab_16nodes_bs16k.yaml" sbatch  -N 16 --job-name=madonna-image-ab-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
# CONFIG_NAME="configs/imagenet/vit/ab_32nodes_bs32k.yaml" sbatch  -N 32 --job-name=madonna-image-ab-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch

# # # resnet strong scaling
# # CONFIG_NAME="configs/imagenet/resnet/ab_2nodes_bs2k0.01.yaml" sbatch -N 2 --job-name=madonna-image-ab-res-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
# CONFIG_NAME="configs/imagenet/resnet/ab_4nodes_bs4k0.01.yaml" sbatch  -N 4 --job-name=madonna-image-ab-res-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
# CONFIG_NAME="configs/imagenet/resnet/ab_8nodes_bs8k0.01.yaml" sbatch  -N 8 --job-name=madonna-image-ab-res-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
# CONFIG_NAME="configs/imagenet/resnet/ab_16nodes_bs16k0.01.yaml" sbatch  -N 16 --job-name=madonna-image-ab-res-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
# CONFIG_NAME="configs/imagenet/resnet/ab_32nodes_bs32k0.01.yaml" sbatch  -N 32 --job-name=madonna-image-ab-res-nogroup /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch

CONFIG_NAME="configs/cifar10/vgg16/ab_2nodes.yaml" sbatch  -N 2 --job-name=madonna-cifar-vgg19 /hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/scripts/launch_singularity.sbatch
