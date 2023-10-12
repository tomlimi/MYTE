#!/bin/bash

# This script is used to run Morfessor on a corpus of text.

LANGUAGES="eng_Latn spa_Latn por_Latn fra_Latn ita_Latn ron_Latn pol_Latn mlt_Latn heb_Hebr arb_Arab jpn_Jpan kor_Hang tel_Telu tam_Taml bod_Tibt sin_Sinh"
BYTE_PATCHES="24 36 48"
MORPH_TYP_NUM="4096 16384 65536 0"



for BP in $BYTE_PATCHES; do
    LEXICON_FILE="morfessor_in/joint_bp_${BP}_dev.lex"
    for MTN in $MORPH_TYP_NUM; do
        EXPERIMENT_NAME="bp_${BP}_ml_0_mtn_${MTN}"
        OUTPUT_MODEL="morfessor_out/joint_model_${EXPERIMENT_NAME}.bin"
        if [ "$MTN" != 0 ]; then
          morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" -s "${OUTPUT_MODEL}" --atom-separator '-' --max-epochs 20  --num-morph-types "${MTN}"
        else
          morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" -s "${OUTPUT_MODEL}" --atom-separator '-' --max-epochs 20
        fi

        for LANG in $LANGUAGES; do
            echo "Running Segmentation on ${LANG} with byte patches ${BP}, morph type number ${MTN}."
            source morfessor-eval-lang.sh "${LANG}" "dev" "${BP}" "0" "${MTN}" "1"
            source morfessor-eval-lang.sh "${LANG}" "devtest" "${BP}" "0" "${MTN}" "1"
            done
        done
done