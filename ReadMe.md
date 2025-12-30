# Setup
conda activate
conda create --name pub_viruses python=3.10
conda config --add channels conda-forge
conda config --set channel_priority strict

conda activate pub_viruses

conda install numpy pandas 

# Download SAM 2

SAM2 requires python>=3.10, torch>-2.5.1, torchvision>=0.20.1

git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
mkdir -p /checkpoints/
wget -P /checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt