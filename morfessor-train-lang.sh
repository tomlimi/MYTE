#!/bin/bash

# This script is used to run Morfessor on a corpus of text.

LANG_NAME=$1
SPLIT=$2
BYTE_PATCHES=$3
MORPH_LENGTH=$4
MORPH_TYP_NUM=$5



DIR_NAME="flores200_dataset/${SPLIT}"
FILE_NAME="${DIR_NAME}/${LANG_NAME}.${SPLIT}"



LEXICON_DIR="morfessor_in"
EXPERIMENT_NAME="bp_${BYTE_PATCHES}_ml_${MORPH_LENGTH}_mtn_${MORPH_TYP_NUM}"
LEXICON_FILE="${LEXICON_DIR}/${LANG_NAME}/bp_${BYTE_PATCHES}_${SPLIT}.lex"


CORPUS_FILE="${LEXICON_DIR}/${LANG_NAME}/corpus_${SPLIT}.txt"

mkdir -p "${LEXICON_DIR}/${LANG_NAME}"

OUTPUT_DIR="morfessor_out"
OUTPUT_MODEL="${OUTPUT_DIR}/${LANG_NAME}/model_${EXPERIMENT_NAME}.bin"
OUTPUT_FILE="${OUTPUT_DIR}/${LANG_NAME}/dev_${EXPERIMENT_NAME}_${SPLIT}.seg"

mkdir -p "${OUTPUT_DIR}/${LANG_NAME}"

# Prepare the lexicon for morfessor.
if [ "$BYTE_PATCHES" = 0 ] ; then
    python prepare_corpus_lexicon.py "${FILE_NAME}" "${CORPUS_FILE}" --lexicon "${LEXICON_FILE}"
else
    python prepare_corpus_lexicon.py "${FILE_NAME}" "${CORPUS_FILE}"  --lexicon "${LEXICON_FILE}" --byte_patches --patch_size "${BYTE_PATCHES}"
fi



# Run Morfessor on the corpus.
if [ "$MORPH_LENGTH" != 0 ] ; then
    morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" -s "${OUTPUT_MODEL}" --atom-separator '-' --max-epochs 10  --morph-length "${MORPH_LENGTH}"
elif [ "$MORPH_TYP_NUM" != 0 ]; then
    morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" -s "${OUTPUT_MODEL}" --atom-separator '-' --max-epochs 10  --num-morph-types "${MORPH_TYP_NUM}"
else
    morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" -s "${OUTPUT_MODEL}" --atom-separator '-' --max-epochs 10
fi
