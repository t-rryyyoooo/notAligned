#!/bin/bash

#Input
readonly TEXT="$HOME/Desktop/data/textList"
readonly WEIGHT="$HOME/Desktop/data/modelweight"
readonly HISTORY="$HOME/Desktop/data/history"
readonly LOG="$HOME/Desktop/data/log"
#readonly NUMBERS=(001 017 020 022 043 082 094 115 120 137 173 174 205 019 023 054 093 096 123 127 136 141 153 188 191 201)
readonly NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)
readonly DATA="$HOME/Desktop/data/kits19"
readonly SAVE="$HOME/Desktop/data/patch"
readonly BATCHSIZE=15
readonly EPOCH=300
readonly RESULT="$HOME/Desktop/data/result"
readonly PATCHSIZE="256-256-5"

# Determine input details.
echo -n DIRECTORY_NAME:
read directory
echo -n "Is subdirectory name original?[yes/no]:"
read choice

while [ ! $choice = "yes" -a ! $choice = "no" ]
do

echo -n "Is subdirectory name original?[yes/no]:"
read choice

done

if [ $choice = "yes" ]; then
  sub="original"

else
  echo -n SUBDIRECTORY_NAME:
  read sub

fi

training="${TEXT}/${directory}/${sub}/training.txt"
validation="${TEXT}/${directory}/${sub}/validation.txt"
initial="${WEIGHT}/${directory}/${sub}/initial.hdf5"
best="${WEIGHT}/${directory}/${sub}/best.hdf5"
latest="${WEIGHT}/${directory}/${sub}/latest.hdf5"
histories="${HISTORY}/${directory}/${sub}/loss.txt"
log="${LOG}/${directory}/${sub}"

echo -n GPU_ID:
read id

echo -n "Which weight do you select?[best/latest]:"
read whichWeight

while [ ! $whichWeight = "best" -a ! $whichWeight = "latest" ]
do

echo -n "Which weight do you select?[best/latest]:"
read whichWeight

done


  if [ $whichWeight = "best" ];then
    segWeight="${best}"
  elif [ $whichWeight = "latest" ];then
    segWeight="${latest}"

  fi

echo "---Training---"
echo "Training:${training}"
echo "Validation:${validation}"
echo "InitialWeight:${initial}"
echo "BestWeight:${best}"
echo "LatestWeight:${latest}"
echo "Loss history:${histories}"
echo "Log:${log}"

# Training module.
#python3 buildUnet.py ${training} --bestfile ${best} --initialfile ${initial} --latestfile ${latest} -t ${validation} --history ${histories} -b ${BATCHSIZE} -e ${EPOCH} -g ${id} --logdir ${log}

# Segmentation module.
echo "---Segmentation---"
save="${SAVE}/${directory}/segmentation/${sub}"
for number in ${NUMBERS[@]}
do
  savePat="${save}/case_00${number}/label.mha"
  imageDirectory="${DATA}/case_00${number}"
  
  echo "ImageDirectory:${imageDirectory}"
  echo "Weight:${segWeight}"
  echo "Save:${savePat}"
  echo "GPU_ID:${id}"

  python3 segmentation.py ${imageDirectory} ${segWeight} ${savePat} -g ${id} --noFlip --outputImageSize ${PATCHSIZE}

done

# Caluculate DICE module.
echo "---CaluculateDICE---"

result="${RESULT}/${directory}/${sub}"

mkdir -p ${result}

results="${result}/DICE.txt"

echo "True:${DATA}"
echo "Predict:${save}"
echo "ResultText:${results}"

python3 caluculateDICE.py ${DATA} ${save} > ${results}

if [ $? -eq 0 ]; then
  echo "Done."
  echo ${save} >> ${results}

else
  echo "Fail"

fi
