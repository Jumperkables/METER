#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-8]
#SBATCH --qos short
#SBATCH --job-name vqa_mao-9_normonly-expanded-nc-pt4_avsc_METER
#SBATCH --mem 20G
#SBATCH --gres gpu:1
#SBATCH -o ../../../results/vqa_mao-9_normonly-expanded-nc-pt4_avsc_METER.out

cd ../../..
source venv/bin/activate

python run.py with norm_clipping=0.4 normonly_flag=expanded data_root=data/vqa_Expanded-nc-gt0.4_occ_gte9_arrow num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=200 load_path=checkpoints/meter_clip16_224_roberta_pretrain.ckpt clip16 text_roberta image_size=224 clip_randaug num_workers=4 loss_type=avsc vqav2_label_size=392
