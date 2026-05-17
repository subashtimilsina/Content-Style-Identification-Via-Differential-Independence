HPC_SHARE=/nfs/stak/users/timilsis/hpc-share/content_style_demo

# Download AFHQ dataset
URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
ZIP_FILE=$HPC_SHARE/data/afhq.zip
mkdir -p $HPC_SHARE/data
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d $HPC_SHARE/data
rm $ZIP_FILE

# Download CelebA-HQ dataset
URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
ZIP_FILE=$HPC_SHARE/data/celeba_hq.zip
mkdir -p $HPC_SHARE/data
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d $HPC_SHARE/data
rm $ZIP_FILE