#!/bin/bash
cd ..
source venv/bin/activate

python run.py with data_root=data/vqa_arrow num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=4 load_path=checkpoints/meter_clip16_224_roberta_pretrain.ckpt clip16 text_roberta image_size=224 clip_randaug 
