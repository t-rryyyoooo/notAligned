#!/bin/bash

# Input
WEIGHT="$HOME/Desktop/data/modelweight"

echo -n which machine do you use?[gv100/red]:
read machine

while [ ! $machine = "gv100" -a ! $machine = "red" ]
do
  
  echo -n which machine do you use?[gv100/red]:
  read machine

done

if [ $machine = "gv100" ];then
  ip="192.168.1.210"

elif [ $machine = "red" ];then
  ip="192.168.1.208"

fi

echo -n DIRECTORY:
read directory

echo -n "Is subdirectory original?[yes/no]:"
read choice 

while [ ! $choice = "yes" -a ! $choice = "no" ]
do
  
  echo -n "Is subdirectory original?[yes/no]:"
  read choice 

done



if [ $choice = "yes" ];then
  sub="original"

elif [ $choice = "no" ];then
  echo -n SUB:
  read sub

fi

weight="${WEIGHT}/${directory}/${sub}"

mkdir -p ${weight}

latest="${weight}/latest.hdf5"
best="${weight}/best.hdf5"
initial="${weight}/initial.hdf5"

scp -P 22222 "${ip}:${latest}" ${latest}
scp -P 22222 "${ip}:${best}" ${best}
scp -P 22222 "${ip}:${initial}" ${initial}

echo "From ${ip}:${latest}" 
echo "To ${latest}"
echo "From ${ip}:${best}" 
echo "To ${best}"
echo "From ${ip}:${initial}" 
echo "To ${initial}"

