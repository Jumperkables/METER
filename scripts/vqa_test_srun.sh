#!/bin/bash
srun -N 1 -p res-gpu-small -c 4 -t 2-00:00 --qos short --job-name test --mem 12G --gres gpu:1 vqa_test.sh
