#!/bin/bash

for ((i = 0; i < 100; i += 1));
do
  mkdir $i;
  cd $i;
  sbatch --export=JB=$i ../mpg.sh
  sleep 7
  cd ../;
done
