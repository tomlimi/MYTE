#!/bin/bash

# This script is used to run Morfessor on a corpus of text.

LANGUAGES="eng_Latn spa_Latn por_Latn fra_Latn ita_Latn ron_Latn pol_Latn mlt_Latn heb_Hebr arb_Arab jpn_Jpan kor_Hang tel_Telu tam_Taml bod_Tibt sin_Sinh"
BYTE_PATCHES="48 12 24 36"
MORPH_TYP_NUM="4096 1024 256 0"
#MORPH_LENGTH="2.0 4.0 6.0 8.0"


for BP in $BYTE_PATCHES; do
    for MTN in $MORPH_TYP_NUM; do
        for LANG in $LANGUAGES; do
            echo "Running Morfessor on ${LANG} with byte patches ${BP}, morph type number ${MTN}."
            source morfessor-train-lang.sh "${LANG}" "dev" "${BP}" "0" "${MTN}"
            source morfessor-eval-lang.sh "${LANG}" "dev" "${BP}" "0" "${MTN}" "0"
            source morfessor-eval-lang.sh "${LANG}" "devtest" "${BP}" "0" "${MTN}" "0"
            done
        done
#    for ML in $MORPH_LENGTH; do
#        for LANG in $LANGUAGES; do
#            source morfessor-train-lang.sh "${LANG}" "dev" "${BP}" "${ML}" "0"
#            source morfessor-eval-lang.sh "${LANG}" "dev" "${BP}" "${ML}" "0"
#            source morfessor-eval-lang.sh "${LANG}" "devtest" "${BP}" "${ML}" "0"
#            done
#        done
    done
