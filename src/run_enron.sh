#!/bin/bash
#
#BSUB -J alg2
#BSUB -n 16
#BSUB -R "span[ptile=16]"
#BSUB -q gpu
#BSUB -e %J.error
#BSUB -o %J.output

./revmax -g data/fb.bin -b 0.02 -q -G -e 0.1 >fb.csv
./revmax -g data/fb.bin -b 0.02 -q -Q -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.04 -q -G -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.04 -q -Q -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.06 -q -G -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.06 -q -Q -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.08 -q -G -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.08 -q -Q -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.1 -q -G -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.1 -q -Q -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.2 -q -G -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.2 -q -Q -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.3 -q -G -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.3 -q -Q -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.4 -q -G -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.4 -q -Q -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.5 -q -G -e 0.1 >>fb.csv
./revmax -g data/fb.bin -b 0.5 -q -Q -e 0.1 >>fb.csv
