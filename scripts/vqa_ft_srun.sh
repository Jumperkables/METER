#!/bin/bash
srun -N 1 -p res-gpu-small -c 4 -t 2-00:00 -x gpu[0-8] --qos short --job-name test --mem 16G --gres gpu:1 vqa_ft.sh
