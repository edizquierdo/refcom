#!/bin/bash

for ((i=0; i<100; i+=1));
do
        echo $i 
        tail -3 $i/evol.dat
done
