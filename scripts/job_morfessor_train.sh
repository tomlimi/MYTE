#! /bin/bash
#SBATCH --job-name=train-morfessor
#SBATCH --output=slurm_output/train-morfessor-%A-%a.log
#SBATCH --error=slurm_output/train-morfessor-%A-%a.log
#SBATCH --account=zlab
#SBATCH --partition=ckpt
#SBATCH --time=22:59:00
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=16G

cd /gscratch/zlab/tomlim/mseg/fair_segmentation/src || exit
source ../../mseg/bin/activate

shopt -s nullglob

LANGUAGES=('af' 'am' 'az' 'be' 'bg' 'bn' 'ca' 'ceb' 'co' 'cs' 'cy' 'da' 'de' 'el' 'eo' 'et' 'eu' 'fa' 'fi' 'fy' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'hi' 'hmn' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'iw' 'jv' 'ka' 'kk' 'km' 'kn' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'ps' 'ru' 'sd' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'sr' 'st' 'su' 'sv' 'sw' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu')
LANGUAGES=('af' 'am' 'az' 'be' 'bg' 'bn' 'ca' 'ceb' 'co' 'cs' 'cy' 'da' 'de' 'el' 'eo' 'et' 'eu' 'fa' 'fi' 'fy' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'hi' 'hmn' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'iw' 'jv' 'ka' 'kk' 'km' 'kn' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'ps' 'ru' 'sd' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'sr' 'st' 'su' 'sv' 'sw' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu' 'en' 'es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt' 'he' 'ar' 'ja' 'ko' 'te' 'ta' 'si')
# LANGUAGES=('ar' 'bo' 'et'  'fa' 'haw' 'he' 'hi' 'ja' 'ko' 'lo' 'lt' 'mi'  'ny' 'sm' 'st' 'ta' 'xh')
#LANGUAGES=('en' 'es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt' 'he' 'ar' 'ja' 'ko' 'te' 'ta' 'bo' 'si')
#LANGUAGES=('ceb' 'ru' 'sr' 'sv' 'uk' 'zh')
#LANGUAGES=('lo' )
LANG=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}

LEXICON_FILE="../../lexicons_decomposed_filtered/${LANG}_lex.txt"
MORPH_TYP_NUM=5000
OUTPUT_MODEL="../../morfessor_models_decomposed_filtered/${LANG}_${MORPH_TYP_NUM}.bin"

echo "Training morfessor for ${LANG} with aim of ${MORPH_TYP_NUM} morph types"

if [[ "$MORPH_TYP_NUM" = 0 ]]
then
	morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" -s "${OUTPUT_MODEL}" --atom-separator ' ' --max-epochs 50
else
	morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" -s "${OUTPUT_MODEL}" --atom-separator ' ' --max-epochs 50  --num-morph-types "${MORPH_TYP_NUM}"
fi
