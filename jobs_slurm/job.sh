#!/bin/bash
#SBATCH --partition=gpu-v100 
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00

module load python/anaconda3-2019.10-tensorflowgpu
module load cuda/11.3.0

source activate /home/raissa.souzadeandrad/venv

python travel_main.py -o /home/raissa.souzadeandrad/FeTs_challenge/cpt_sasc -t 2 -e 1.0