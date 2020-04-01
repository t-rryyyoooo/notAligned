#!/bin/bash

#Input
readonly DATA="$HOME/Desktop/data/kits19"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="$HOME/Desktop/data/patch/256_256_3_cancer_noFlip/segmentation"
readonly WEIGHT="$HOME/Desktop/data/modelweight/256_256_3_cancer_noFlip_latest.hdf5"

NUMBERS=(001 017 020 022 043 082 094 115 120 137 173 174 205)
#NUMBERS=(019 023 054 093 096 123 127 136 141 153 188 191 201)
#NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)
NUMBERS=(001 017 020 022 043 082 094 115 120 137 173 174 205 019 023 054 093 096 123 127 136 141 153 188 191 201)

echo -n GPU_ID:
read id
for number in ${NUMBERS[@]}
do


    save="${SAVE}/case_00${number}/label.mha"
    imageDirectory="${DATA}/case_00${number}"

    echo $imageDirectory
    echo $WEIGHT
    echo $save
    echo "GPU ID: $id"

    python3 segmentation.py $imageDirectory $WEIGHT $save -g $id --noFlip --outputImageSize 256-256-3


done
