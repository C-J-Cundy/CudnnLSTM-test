#!/bin/bash

OUTFILE='./answers.txt'
VALFILE='./validation'
rm $OUTFILE
INFILE='in_list'
READINFILE='out_file'

for f in ../../validation/*.wav
do
    python next.py $f
done
