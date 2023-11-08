#! /bin/bash
#SBATCH --job-name=train-unigram
#SBATCH --output=slurm_output/train-unigram-%A-%a.log
#SBATCH --error=slurm_output/train-unigram-%A-%a.log
#SBATCH --account=zlab
#SBATCH --partition=gpu-rtx6k
#SBATCH --time=22:59:00
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=16G


cd /gscratch/zlab/tomlim/mseg/fair_segmentation/src || exit
source ../../mseg/bin/activate

shopt -s nullglob

LANGUAGES=('af' 'am' 'az' 'be' 'bg' 'bn' 'ca' 'ceb' 'co' 'cs' 'cy' 'da' 'de' 'el' 'eo' 'et' 'eu' 'fa' 'fi' 'fy' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'hi' 'hmn' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'iw' 'jv' 'ka' 'kk' 'km' 'kn' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'ps' 'ru' 'sd' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'sr' 'st' 'su' 'sv' 'sw' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu')
LANGUAGES=('af' 'am' 'az' 'be' 'bg' 'bn' 'ca' 'ceb' 'co' 'cs' 'cy' 'da' 'de' 'el' 'eo' 'et' 'eu' 'fa' 'fi' 'fy' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'hi' 'hmn' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'iw' 'jv' 'ka' 'kk' 'km' 'kn' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'ps' 'ru' 'sd' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'sr' 'st' 'su' 'sv' 'sw' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu' 'en' 'es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt' 'he' 'ar' 'ja' 'ko' 'te' 'ta' 'si')
LANG=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}

LEXICON_FILE="../../lexicons_decomposed_filtered/${LANG}_lex.txt"
VOCAB_SIZE=4096
OUTPUT_DIR="../../tokenizers_decomposed_filtered"

echo "Training Unigram tokenizer for ${LANG} with vocab size: ${VOACB_SIZE}"

python train_subword.py --lexicon ${LEXICON_FILE} --out_dir ${OUTPUT_DIR} --language ${LANG} --vocab_size ${VOCAB_SIZE} --type "unigram"
