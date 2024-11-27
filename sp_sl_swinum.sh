#!/bin/bash
#SBATCH --gres=gpu:1      
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10 
#SBATCH --mem=64000M       
#SBATCH --time=23:55:59     
#SBATCH --account=rrg-pfieguth
#SBATCH --mail-type=ALL

module load StdEnv/2020
module load python/3.8.10 cuda cudnn opencv


# Prepare virtualenv

cp -r /home/ashleyc/projects/def-pfieguth/ashleyc/supertransformer $SLURM_TMPDIR/

tar -xf /home/ashleyc/envs/'uda.tar 1.gz' -C $SLURM_TMPDIR/
sed -i "s|/home/j97park/uda/bin/python|$SLURM_TMPDIR/uda/bin/python|g" $SLURM_TMPDIR/uda/bin/*
module unload python

source $SLURM_TMPDIR/uda/bin/activate
export PATH=$SLURM_TMPDIR/uda/bin:$PATH
export PYTHONNOUSERSITE=1

echo "Using Python from: $(which python)"
python --version
pip list

echo "virtual env connected"

# Prepare data
tar -xvf /home/ashleyc/scratch/data/DUTS-1024-v3.tar.xz -C $SLURM_TMPDIR

logdir=/home/ashleyc/projects/def-pfieguth/ashleyc/logs/

# Start training
cd $SLURM_TMPDIR/supertransformer/

git checkout sp-swinum-H1-R0-C1-M0

python train.py \
    --model SP_SWINUM \
    --dataloader SPFFFT \
    --dataset_tr $SLURM_TMPDIR/DUTS/DUTS-TR/ \
    --dataset_test $SLURM_TMPDIR/DUTS/DUTS-TE/ \
    --eval \
    --batch_size 32 \
    --epoch 400 \
    --num_workers 10 \
    --val_freq 1.0 \
    --es_patience 50 \
    --lr 0.001 \
    --logdir ${logdir} \
    --num_seg 1024 \
    --coeff 10 \
    --tag $SLURM_JOB_ID \
    --dilation 1 \
    --compactness 10 \
    --seed 42 \
    --size 448 \
    --dropout_edge 0 \
    --heads 2 4 8 \
    --dims 32 64 128 \
    --depths 2 2 6 \
    --window_size 8 \
    --warmup_epochs 0 \
    --drop_path 0.1 \
    --mlp_ratio 2
    #--pretrain /home/j97park/projects/def-pfieguth/j97park/sp_imgnet_ogswin_ape_51851743.ckpt
    ##--swin_factor $4
    #--pretrain /home/j97park/projects/def-pfieguth/j97park/image_imgnet_acc80.ckpt
    #--kernels 4 4 4 \
    #--window_size 4
    