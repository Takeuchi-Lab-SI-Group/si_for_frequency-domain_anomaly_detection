#!/bin/bash

for window in 1024 512; do
    for delta in 0.04 0.08 0.12 0.16; do
        python experiment_tpr.py $window $delta
    done
done