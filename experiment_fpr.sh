#!/bin/bash

for window in 1024 512; do
    for T in 40 60 80 100; do
        python experiment_fpr.py $window $T
    done
done