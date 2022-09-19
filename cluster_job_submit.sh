#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12 
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```


# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
# SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
# SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
# SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
# SBATCH --mem=16000

# Number of CPUs to use. Check `cluster-status` for node configurations
# SBATCH --cpus-per-task=24

# Maximum time for the job to run, format: days-hours:minutes:seconds
# SBATCH --time=04:00:00

# Recommend setto certain node so we don't waste time copy dataset
# SBATCH --nodelist=landonia02

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch_big
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=centerpoint-kitti
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
repo_home=/home/${USER}/CenterPoint-KITTI
src_path=/home/${USER}/dataset/view_of_delft_PUBLIC

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/view_of_delft_PUBLIC
mkdir -p ${dest_path}  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

cd $repo_home
#ln -s ${dest_path}/lidar ${repo_home}/data/vod_lidar
#ln -s ${dest_path}/radar ${repo_home}/data/vod_radar
# python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/vod_lidar_dataset.yaml
# python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/vod_radar_dataset.yaml
# python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/vod_radar_dataset_iassd_car.yaml
cd ./tools
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-vod-radar.yaml --epoch 80 --workers 8 --extra_tag points1024radar_scaleSA --batch_size 16 --eval_save True --eval_epoch 2
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-GAN-vod-aug.yaml --epoch 40 --workers 8 --extra_tag points1024-scaleSA --batch_size 8 --eval_save True --eval_epoch 1 --pretrained_model ../output/IA-SSD-vod-radar/iassd_best_aug_new/checkpoint_epoch_36.pth
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-GAN-vod-lidar.yaml --epoch 40 --workers 8 --extra_tag nofeataug_freeze_head --batch_size 8 --eval_save True --eval_epoch 1 --pretrained_model ../output/IA-SSD-vod-lidar/all_cls/ckpt/checkpoint_epoch_80.pth
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-vod-lidar-freeze.yaml --epoch 40 --workers 8 --extra_tag freeze_backbone --batch_size 8 --eval_save True --eval_epoch 1 --pretrained_model ../output/IA-SSD-vod-lidar/all_cls/ckpt/checkpoint_epoch_80.pth
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-GAN-vod-aug-car.yaml --epoch 80 --workers 8 --extra_tag car_only --batch_size 8 --eval_save True --eval_epoch 1
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-vod-radar-car.yaml --epoch 80 --workers 8 --extra_tag car_only --batch_size 8 --eval_save True --eval_epoch 1
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-vod-radar-car.yaml --epoch 80 --workers 8 --extra_tag car_only_xavier --batch_size 8 --eval_save True --eval_epoch 1

# python train.py --cfg_file cfgs/kitti_models/IA-SSD-GAN-vod-aug-xyz.yaml --epoch 30 --workers 8 --extra_tag ablation --batch_size 8 --eval_save True --eval_epoch 1 --pretrained_model ../output/IA-SSD-vod-radar/iassd_best_aug_new/checkpoint_epoch_36.pth
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-GAN-vod-aug-xyz-r.yaml --epoch 30 --workers 8 --extra_tag ablation --batch_size 8 --eval_save True --eval_epoch 1 --pretrained_model ../output/IA-SSD-vod-radar/iassd_best_aug_new/checkpoint_epoch_36.pth
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-GAN-vod-aug-xyz-r-v.yaml --epoch 30 --workers 8 --extra_tag ablation --batch_size 8 --eval_save True --eval_epoch 1 --pretrained_model ../output/IA-SSD-vod-radar/iassd_best_aug_new/checkpoint_epoch_36.pth
# python train.py --cfg_file cfgs/kitti_models/IA-SSD-GAN-vod-aug-xyz-v.yaml --epoch 30 --workers 8 --extra_tag ablation --batch_size 8 --eval_save True --eval_epoch 1 --pretrained_model ../output/IA-SSD-vod-radar/iassd_best_aug_new/checkpoint_epoch_36.pth

python train.py --cfg_file cfgs/kitti_models/IA-SSD-GAN-vod-aug-best.yaml --epoch 30 --workers 8 --extra_tag new_lr --batch_size 8 --eval_save True --eval_epoch 1 --pretrained_model ../output/IA-SSD-vod-radar/iassd_best_aug_new/checkpoint_epoch_36.pth

# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

# echo "Moving output data back to DFS"

# src_path=${SCRATCH_HOME}/mnist/data/output
# dest_path=${repo_home}/experiments/examples/mnist/data/output
# rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"