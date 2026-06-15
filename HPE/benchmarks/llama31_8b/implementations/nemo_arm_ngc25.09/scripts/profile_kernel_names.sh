#!/bin/bash

INPUTDIR=$1

for file in ${INPUTDIR}/*.nsys-rep; do
    TEMPFILE=$(tempfile)
    nsys stats --report cuda_gpu_kern_sum --format csv --output ${TEMPFILE} $file
    python3 -c $'import csv,sys;reader=csv.reader(sys.stdin);\nnext(reader)\nfor row in reader: print(row[-1])' < ${TEMPFILE}_cuda_gpu_kern_sum.csv > ${file%.*}.kernels.txt
    rm ${TEMPFILE}
done
