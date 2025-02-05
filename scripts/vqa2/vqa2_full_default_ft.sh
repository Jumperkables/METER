#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-8]
#SBATCH --qos long-high-prio
#SBATCH --job-name vqa2_full_default_METER
#SBATCH --mem 20G
#SBATCH --gres gpu:1
#SBATCH -o ../../results/vqa2_full_default_METER.out

cd ../..
source venv/bin/activate
echo "Check if the normonly_flag should be true"
#python run.py with normonly_flag=None data_root=data/vqa2_full_arrow num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=200 load_path=checkpoints/meter_clip16_224_roberta_pretrain.ckpt clip16 text_roberta image_size=224 clip_randaug num_workers=4 loss_type=default vqav2_label_size=3128
