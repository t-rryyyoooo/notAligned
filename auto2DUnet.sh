#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_"
readonly WEIGHT="$HOME/Desktop/data/modelweight"
readonly HISTORY="$HOME/Desktop/data/history/history_"
readonly LOG="$HOME/Desktop/data/log"

echo -n Suffix:
read suffix
echo -n "Is the weight file's prefix the same as above?[yes/no]:"
read choice

training="${TRAINING}${suffix}.txt"
validation="${VALIDATION}${suffix}.txt"

if [ $choice = "yes" ]; then
	histories="${HISTORY}${suffix}.txt"
	initialWeight="${WEIGHT}/${suffix}_initial.hdf5"
        bestWeight="${WEIGHT}/${suffix}_best.hdf5"
	latestWeight="${WEIGHT}/${suffix}_latest.hdf5"
  log="${LOG}/${suffix}"

else
        echo -n suffix:
        read newSuffix

	histories="${HISTORY}${newSuffix}.txt"
	initialWeight="${WEIGHT}/${newSuffix}_initial.hdf5"
        bestWeight="${WEIGHT}/${newSuffix}_best.hdf5"
	latestWeight="${WEIGHT}/${newSuffix}_latest.hdf5"
  log="${LOG}/${newSuffix}"

fi

echo -n GPU_ID:
read id
echo "Training:${training}"
echo "Validation:${validation}"
echo "InitialWeight:${initialWeight}"
echo "BestWeight:${bestWeight}"
echo "LatestWeight:${latestWeight}"
echo $histories
echo "Log:${log}"

python3 buildUnet.py ${training} --bestfile ${bestWeight} --initialfile ${initialWeight} --latestfile ${latestWeight} -t ${validation} --history ${histories} -b 15 -e 300 -g $id --logdir ${log}

