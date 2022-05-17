#!/bin/bash
#SBATCH --job-name=comparison
#SBATCH --time=12:00:00
#SBATCH --mem=20GB
#SBATCH --open-mode=append
#SBATCH --output=/home/sj2539/final-project-group_3/%j_%x.out
#SBATCH --error=/home/sj2539/final-project-group_3/%j_%x.err
#SBATCH --export=ALL
#SBATCH --requeue


## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.

singularity exec --overlay /scratch/sj2539/comparison/comparison.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "

source /ext3/env.sh
python /home/sj2539/final-project-group_3/comparison.py
"
