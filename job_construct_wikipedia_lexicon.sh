#! /bin/bash
#SBATCH --job-name=construct-wiki-lexicon
#SBATCH --output=construct-wiki-lexicon-%x-%A-%a.log
#SBATCH --error=construct-wiki-lexicon-%x-%A-%a.log
#SBATCH --account=zlab
#SBATCH --time=12:59:00
#SBATCH --ntasks=1
#SBATCH --c 5
#SBATCH --mem=32G
#SBATCH --workdir /usr/lusers/tomlim/my_gscratch/mseg/fair_segmentation/src

source ../../mseg/bin/activate


shopt -s nullglob

LANGUAGES=('en' 'es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt' 'he' 'ar' 'ja' 'ko' 'te' 'ta' 'bo' 'si')
LATN_LANGUAGES=('en' 'es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt')
LANG=${LANGUGES[$SLURM_ARRAY_TASK_ID]}
IF_LATN=(( $SLURM_ARRAY_TASK_ID <= ${#LATN_LANGUAGES[@]} ))

echo "Creating wikipedia corpus for ${LANG}"
echo "If Latin: ${IF_LATN}"

if [ $IF_LATN ]
then
    python construct_wikipedia_lexicon.py --lang $LANG --lexicon_directory "../../lexicons"  --do_capitalize
else
    python construct_wikipedia_lexicon.py --lang $LANG --lexicon_directory "../../lexicons"
fi