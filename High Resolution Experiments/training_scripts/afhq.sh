export CUDA_HOME=/usr/local/apps/cuda/12.1
export CC=/usr/local/apps/gcc/11.5/bin/gcc
export CXX=/usr/local/apps/gcc/11.5/bin/g++
export CUDA_VISIBLE_DEVICES=0

RUN_NAME=afhq_v2
HPC_SHARE=/nfs/stak/users/timilsis/hpc-share/content_style_demo

python train.py --outdir=$HPC_SHARE/training_runs_$RUN_NAME \
                --data=$HPC_SHARE/data/$RUN_NAME \
                --gpus=1 \
                --map_type=fixed \
                --num_c_res=5 \
                --i_dim=8 \
                --sparse_weight=0.3 \
                --cfg=mlp3 \
                --style_mixing_prob=0.0 \
                --metrics=pr50k3_full,fid50k_full \
                --cond=true \
                --wandb_run_name=$RUN_NAME;