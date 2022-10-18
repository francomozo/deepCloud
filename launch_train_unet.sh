#!/bin/bash
#SBATCH --job-name=R2
#SBATCH --ntasks=2
#SABTCH --mem=3040
#SBATCH --time=5-00:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=scripts/paper/R2.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/deepCloud
python train_unet.py --dataset 'uru' --epochs 100 --batch_size 2 --architecture 'R2' --init_filters 64 --time_horizon '60min' --testing_loop False

