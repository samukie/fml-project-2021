#!/bin/bash
source ~/.bashrc	

# call this script e.g. like so:
# $ ./slurmgo.sh nameOfLatestConfig 0-06:00:00 64000 gpushort 


# script call signature:

name=$1
duration=${2:-"0-12:00:00"} # duration should have syntax 2-12:15:59
memory=${3:-"128000"}
partition=${4:-"students"}
template_name=${5:-"template"}

echo $template_name

clear

echo "Starting procedure for job $name ..."
echo


#paths
sbatch_path=sbatch/

#extensions
sbatch_ext=".sh"


sbatch="$sbatch_path$name$sbatch_ext"
echo "creating sbatch $sbatch:"

template="${sbatch_path}$template_name$sbatch_ext"
cp -rp "$template" "$sbatch"

#replace all occurences of JOBNAME with name
sed -i "s/JOBNAME/$name/g" $sbatch

#more replacements
sed -i "s/DURATION/$duration/g" $sbatch
sed -i "s/MEMORY/$memory/g" $sbatch
sed -i "s/PARTITION/$partition/g" $sbatch

echo
echo "---------------------------------------------------------------------------"
cat $sbatch
echo "---------------------------------------------------------------------------"
echo 

# execute sbatch:
jobnum=$(sbatch $sbatch | awk 'NF>1{print $NF}')
jobinfo=$(squeue | grep $jobnum)

echo
echo "Started Job number $jobnum"
echo
echo "$jobinfo"
echo
echo model can be found under $model_path$model_dir

echo 
echo "==========================================================================="
echo 
echo "    starting training job named $name, now go away and pray"
echo
echo "                                   🛐"
echo
echo "==========================================================================="
squeue | grep 'koss'
echo





