#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --qos short
#SBATCH --job-name 
#SBATCH --mem 12G
#SBATCH --gres gpu:1
#SBATCH -o ../result/test.out
cd ..
source venv/bin/activate

python run.py with data_root=data/vqa_arrow num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=checkpoints/meter_clip16_288_roberta_vqa.ckpt clip16 text_roberta image_size=288 test_only=True
