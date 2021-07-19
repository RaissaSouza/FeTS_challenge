#!/bin/bash
#SBATCH --partition=gpu-v100 
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=5:00:00

module load python/anaconda3-2019.10-tensorflowgpu
module load cuda/11.3.0

source activate /home/raissa.souzadeandrad/venv

python inference.py -d /home/raissa.souzadeandrad/MICCAI_FeTS2021_ValidationData/ -m /home/raissa.souzadeandrad/FeTs_challenge/cpt_cl_1/central_learning.pth -o /home/raissa.souzadeandrad/FeTs_challenge/Inferences/CL_1/