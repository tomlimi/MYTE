#! /bin/bash

shopt -s nullglob

LANGUAGES=('en' 'es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt' 'he' 'ar' 'ja' 'ko' 'te' 'ta' 'bo' 'si')
LANGUAGE_COUNT=$(( ${#LANGUAGES[@]} - 1 ))

sbatch --array=0-${LANGUAGE_COUNT} job_construct_wikipedia_lexicon.sh
