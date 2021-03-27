#!/bin/bash
#
#SBATCH --job-name='convRandPeace'
#SBATCH -n1 -c1
#SBATCH --mem=128000
#SBATCH -p students 
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=0-12:00:00
#SBATCH -o 'srunz/convRandPeace.stdout'
#SBATCH -e 'srunz/convRandPeace.stderr'

printf "$(date -R)\t[start training convRandPeace]\n" >> time.log
source ~/.bashrc
srun python3.8 main.py "play" "--agents" "actor_critic" "peaceful_agent" "random_agent" "--train" "1" "--no-gui" "--n-rounds" "1000000"
printf "$(date -R)\t[done training convRandPeace]\n" >> time.log

