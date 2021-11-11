#!/bin/sh

#SBATCH -J  multiple      # Job name
#SBATCH -o  ./out/multiplechoice.%j.out   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p 2080ti          # queue  name  or  partiton name
#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

## 노드 지정하지않기
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:4
#SBTACH   --ntasks=4
#SBATCH   --tasks-per-node=4
#SBATCH   --cpus-per-task=2



cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

## path  Erase because of the crash
module  purge
#module  load  postech

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source  $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate QA "
conda activate QA

export PYTHONPATH=.


TRAIN_DIR=$HOME/bert-multiplechoice
python main.py --batch_size 8 --max_length 512 --num_worker 8 --gpus 4

echo " conda deactivate QA "

conda deactivate QA

squeue  --job  $SLURM_JOBID

echo  "##### END #####"
