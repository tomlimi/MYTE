#! /bin/bash
#SBATCH --job-name=flores-lm
#SBATCH --output=slurm_output/flores-lm-%A.log
#SBATCH --error=slurm_output/flores-lm-%A.log
#SBATCH --account=zlab
#SBATCH --gres=gpu:1
#SBATCH--partition=gpu-a40
#SBATCH --time=22:59:00
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=32G

MODEL_TYPE=$1
MODEL_SIZE=$2
TRANSLATE=$3

MODEL_STEPS=250000

cd /gscratch/zlab/tomlim/mseg/fair_segmentation/src || exit
source ../../mseg/bin/activate

if [ "$TRANSLATE" = 1 ]
then
    echo "Translating"
    python flores_modeling.py --model_type $MODEL_TYPE --model_size $MODEL_SIZE --model_steps $MODEL_STEPS --en_translation
else
    echo "Full language modeling"
    python flores_modeling.py --model_type $MODEL_TYPE --model_size $MODEL_SIZE --model_steps $MODEL_STEPS
fi
