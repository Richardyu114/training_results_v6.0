#!/bin/bash

i=1
for f in $(ls *batch*_run*.log | sort -t_ -k2.6,2n -k3.4,3n); do
  echo mv -v "$f" "result_${i}.txt"
  ((i++))
done

