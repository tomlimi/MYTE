#!/bin/bash

# The script to download the MUSE dictionaries

# Download the MUSE dictionaries

LANGUAGES=('af sq ar bs bg ca zh hr cs da nl en et tl fi fr de el he hi hu id it ja ko lv lt mk ms no fa pl pt ro ru sr sk sl es sv ta tr uk vi')
SAVE_DIR="muse"

for lang in $LANGUAGES
do
    wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-${lang}.txt -O ${SAVE_DIR}/${lang}_dir.txt
done

