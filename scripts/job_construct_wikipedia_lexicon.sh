#! /bin/bash
#SBATCH --job-name=construct-wiki-lexicon
#SBATCH --output=slurm_output/construct-wiki-lexicon-%A-%a.log
#SBATCH --error=slurm_output/construct-wiki-lexicon-%A-%a.log
#SBATCH --account=zlab
#SBATCH --partition=gpu-rtx6k
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=24G

cd /gscratch/zlab/tomlim/mseg/fair_segmentation/src || exit 
source ../../mseg/bin/activate


shopt -s nullglob


LANGUAGES=('af' 'am' 'az' 'be' 'bg' 'bn' 'ca' 'ceb' 'co' 'cs' 'cy' 'da' 'de' 'el' 'eo' 'et' 'eu' 'fa' 'fi' 'fy' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'hi' 'hmn' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'iw' 'jv' 'ka' 'kk' 'km' 'kn' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'ps' 'ru' 'sd' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'sr' 'st' 'su' 'sv' 'sw' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu')
# LANGUAGES=('ar' 'bo' 'et'  'fa' 'haw' 'he' 'hi' 'ja' 'ko' 'lo' 'lt' 'mi'  'ny' 'sm' 'st' 'ta' 'xh')
# LANGUAGES=('ceb' 'ru' 'sr' 'sv' 'uk' 'zh')
# LANGUAGES=('es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt' 'he' 'ar' 'ja' 'ko' 'te' 'ta' 'bo' 'si')
LANGUAGES=('sr' )
LANG=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}

echo "Creating wikipedia corpus for ${LANG}"

sleep $((100 * ${SLURM_ARRAY_TASK_ID}))

python construct_wikipedia_lexicon.py --lang $LANG --lexicon_directory "../../lexicons_filtered" --pre_processing_file "../../byte_maps/decompose_lc.json"  --do_capitalize --min_occurances 0 --lexicon_size 30000 --filter_en
