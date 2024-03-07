#! /bin/bash
#SBATCH --job-name=ft_spelling_correction
#SBATCH --output=slurm_output/ft_spelling_correction-%A.log
#SBATCH --error=slurm_output/ft_spelling_correction-%A.log
#SBATCH --account=zlab
#SBATCH --gres=gpu:1
#SBATCH--partition=gpu-a40
#SBATCH --time=22:59:00
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=64G

MODEL_TYPE=$1
MODEL_SIZE=$2

MODEL_STEPS=250000

cd /gscratch/zlab/tomlim/mseg/fair_segmentation/src || exit
source ../../mseg/bin/activate

python ft_generative_task.py --task "spelling_correction" --directory "../ml-olympiad-multilingual-spell-correction" \
                             --model_type $MODEL_TYPE --model_size $MODEL_SIZE --model_steps $MODEL_STEPS --patience 1