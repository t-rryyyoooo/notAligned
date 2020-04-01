#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/kits19"
readonly TEXT="$HOME/Desktop/kidney/alignedPerPatch/result"
readonly RESULT="$HOME/Desktop/data/patch/256_256_3_cancer/segmentation_sqrtWeighted"
readonly PREFIX="256_256_3_sqrtWeighted"


mkdir -p $TEXT
text="${TEXT}/${PREFIX}.txt"

echo ${TRUE}
echo $RESULT
echo $text

python3 caluculateDICE.py ${TRUE} ${RESULT} > $text
if [ $? -eq 0 ]; then
 echo "Done."
 echo ${RESULT} >> $text

else
 echo "Fail"

fi


