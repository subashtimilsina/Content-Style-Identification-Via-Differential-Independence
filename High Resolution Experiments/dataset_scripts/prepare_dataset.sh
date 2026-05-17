HPC_SHARE=/nfs/stak/users/timilsis/hpc-share/content_style_demo

# Prepare AFHQ dataset
python dataset_tool.py --source=$HPC_SHARE/data/afhq/train/ --dest=$HPC_SHARE/data/afhq_v2 --height=128 --width=128
python downsample_images.py \
    --in_root  $HPC_SHARE/data/afhq \
    --out_root $HPC_SHARE/data/afhq_128 \
    --size 128 --workers 8

# Prepare CelebA-HQ dataset
python dataset_tool.py --source=$HPC_SHARE/data/celeba_hq/train/ --dest=$HPC_SHARE/data/celeba_hq_v2 --height=128 --width=128
python downsample_images.py \
    --in_root  $HPC_SHARE/data/celeba_hq \
    --out_root $HPC_SHARE/data/celeba_hq_128 \
    --size 128 --workers 8