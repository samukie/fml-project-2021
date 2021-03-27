#!/bin/bash
#
#SBATCH --job-name='JOBNAME'
#SBATCH -n1 -c1
#SBATCH --mem=MEMORY
#SBATCH -p PARTITION 
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=DURATION
#SBATCH -o 'srunz/JOBNAME.stdout'
#SBATCH -e 'srunz/JOBNAME.stderr'

printf "$(date -R)\t[start training JOBNAME]\n" >> time.log
source ~/.bashrc
srun python3.8 main.py "play" "--agents" "actor_critic" "peaceful_agent" "random_agent" "--train" "1" "--no-gui" "--n-rounds" "1000000"
printf "$(date -R)\t[done training JOBNAME]\n" >> time.log

