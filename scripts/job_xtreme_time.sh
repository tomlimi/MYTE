#! /bin/bash
#SBATCH --job-name=xtreme-up-time
#SBATCH --output=slurm_output/xtreme-up-time-%A.log
#SBATCH --error=slurm_output/xtreme-up-time-%A.log
#SBATCH --account=zlab
#SBATCH --gres=gpu:1
#SBATCH--partition=gpu-a40
#SBATCH --time=22:59:00
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=32G

MODEL_TYPE=$1
MODEL_SIZE=$2

MODEL_STEPS=250000

cd /gscratch/zlab/tomlim/mseg/fair_segmentation/src || exit
source ../../mseg/bin/activate

for MODEL_TYPE in myt5 byt5
do
  for task in qa_in_lang translation semantic_parsing ner
  do
    python xtreme_time.py --model_type $MODEL_TYPE --model_size $MODEL_SIZE --model_steps $MODEL_STEPS --task $task
  done
done