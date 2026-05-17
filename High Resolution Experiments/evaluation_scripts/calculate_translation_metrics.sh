export CUDA_HOME=/usr/local/apps/cuda/12.1
export CC=/usr/local/apps/gcc/11.5/bin/gcc
export CXX=/usr/local/apps/gcc/11.5/bin/g++
export CUDA_VISIBLE_DEVICES=0

HPC_DATASET_DIR=/nfs/stak/users/timilsis/hpc-share/content_style_demo/data


DATASET_PATH=$HPC_DATASET_DIR
ASSETS_DIR=/nfs/stak/users/timilsis/hpc-share/content_style_demo
echo exists


PKL=/nfs/stak/users/timilsis/hpc-share/content_style_demo/training_runs_afhq_v2/00293-afhq_v2-trans-cond-mlp31/network-snapshot-014396.pkl


DATA=afhq_128
RUNID="$(basename "$(dirname "$PKL")" | sed -nE 's/^([0-9]+)-afhq.*/\1/p')"
TASK=afhq

# DATA=celeba_hq_128
# RUNID="$(basename "$(dirname "$PKL")" | sed -nE 's/^([0-9]+)-celebahq.*/\1/p')"
# TASK=celebahq


DATASET_PATH=$DATASET_PATH/$DATA
OUTDIR="$ASSETS_DIR/translation_metrics_${DATA}_${RUNID}"

python compute_translation_metrics.py \
    --model_path $PKL \
    --dataset_path $DATASET_PATH/val \
    --fid_real_path $DATASET_PATH/train \
    --output_dir $OUTDIR \
    --task $TASK \
    --i_dim 12 \
    --num_styles 10;
    # --force_generation ;