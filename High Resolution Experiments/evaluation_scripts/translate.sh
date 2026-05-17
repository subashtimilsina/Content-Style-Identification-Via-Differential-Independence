# -----------------------------
# Environment (HPC)
# -----------------------------
export CUDA_HOME=/usr/local/apps/cuda/12.1
export CC=/usr/local/apps/gcc/11.5/bin/gcc
export CXX=/usr/local/apps/gcc/11.5/bin/g++
export CUDA_VISIBLE_DEVICES=0

#Select few images from the dataset 
DATA_PATH="/nfs/stak/users/timilsis/hpc-share/content_style_demo/data/afhq/translation"

OUTDIR="/nfs/stak/users/timilsis/hpc-share/content_style_demo/transfer_results"

MODEL=/nfs/stak/users/timilsis/hpc-share/content_style_demo/training_runs_afhq_v2/00293-afhq_v2-trans-cond-mlp31/network-snapshot-014396.pkl



python domain_transfer.py \
  --network $MODEL \
  --content $DATA_PATH/cat \
  --style $DATA_PATH/dog \
  --outdir $OUTDIR \
  --source_class 0 --target_class 1 \
  --num-steps 200 --batchsize 32 \
  --grid_n 100 --grid_prefix cs_grid \

#Select few content and style grids from the dataset

# python domain_transfer.py \
#   --network $MODEL \
#   --content $DATA_PATH/cat \
#   --style $DATA_PATH/dog \
#   --outdir $OUTDIR \
#   --source_class 0 --target_class 1 \
#   --num-steps 200 --batchsize 32 \
#   --grid_n 100 --grid_prefix cs_grid --content_idx "1,6,3" --style_idx "4,7,10" 