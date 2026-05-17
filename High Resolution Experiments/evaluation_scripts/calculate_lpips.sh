export CUDA_HOME=/usr/local/apps/cuda/12.1
export CC=/usr/local/apps/gcc/11.5/bin/gcc
export CXX=/usr/local/apps/gcc/11.5/bin/g++
export CUDA_VISIBLE_DEVICES=0

MODEL=/nfs/stak/users/timilsis/hpc-share/content_style_demo/training_runs_afhq_v2/00293-afhq_v2-trans-cond-mlp31/network-snapshot-014396.pkl



python compute_lpips.py --model_path $MODEL