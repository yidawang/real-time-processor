#!/bin/bash

# copy files in dir1 to dir2
dir1=$1
dir2=$2
imagefreq=$3
#dir1=$dir1'*'
count=0

for f in $dir1
do
  cp $f $dir2
  echo "copy $f done"
  sleep $imagefreq
  let count=count+1
done

