#! /bin/bash
#SBATCH --job-name=downstream-eval
#SBATCH --output=slurm_output/downstream-eval-%A.log
#SBATCH --error=slurm_output/downstream-eval-%A.log
#SBATCH --account=zlab
#SBATCH --gres=gpu:1
#SBATCH--partition=gpu-a40
#SBATCH --time=22:59:00
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=32G

TASK=$1
MODEL_TYPE=$2
CROSS=${3:-""}


# INITIALIZE ENVIRONMENT
echo "Initializing environment"

cd /gscratch/zlab/tomlim/mseg/fair_segmentation/src || exit
source ../../mseg/bin/activate

# SET UP VARIABLES
echo "Setting up variables"

BASE_DIR="/gscratch/zlab/tomlim/mseg"
CHECKPOINT_DIR="${BASE_DIR}/hf_checkpoints"

#baseline


if [ "$TASK" == "xnli" ]; then
    LANGUAGES=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh')
elif [ "$TASK" == "xstorycloze" ]; then
    LANGUAGES=('ar' 'en' 'es' 'eu' 'hi' 'id' 'my' 'ru' 'sw' 'te' 'zh')
else
    echo "Task not recognized"
    exit 1
fi

if [ "$CROSS" == "" ]; then
  RESULTS="${BASE_DIR}/results_ds/${MODEL_NAME}/"
  mkdir -p ${RESULTS}

  # RUN EVALUATION
  echo "Running evaluation"
  python3 prompt.py --model_type ${MODEL_TYPE} --output_dir ${RESULTS} --task ${TASK} --eval_lang ${LANGUAGES[@]} --checkpoint_dir ${CHECKPOINT_DIR}
else
  RESULTS="${BASE_DIR}/results_ds/${MODEL_NAME}/"
  mkdir -p ${RESULTS}
  # RUN EVALUATION
  echo "Running evaluation with en demonstrations"
  python3 prompt.py --model_type ${MODEL_TYPE} --output_dir ${RESULTS} --task ${TASK} --eval_lang ${LANGUAGES[@]} --demo_lang "en" --k 8 --checkpoint_dir ${CHECKPOINT_DIR}
fi

echo "Evaluation finished"