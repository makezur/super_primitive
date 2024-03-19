#

mkdir models
wget -O models/scannet.pth https://huggingface.co/datasets/makezur/SuperPrimitive-Data/resolve/main/scannet.pth?download=true
wget -O models/sam_vit_h_4b8939.pth https://huggingface.co/datasets/makezur/SuperPrimitive-Data/resolve/main/sam_vit_h_4b8939.pth?download=true

mkdir datasets
wget -O datasets/replica.zip https://huggingface.co/datasets/makezur/SuperPrimitive-Data/resolve/main/replica_scene.zip?download=true
unzip datasets/replica.zip -d datasets/replica
rm datasets/replica.zip

wget -O datasets/TUM_fr1.zip https://huggingface.co/datasets/makezur/SuperPrimitive-Data/resolve/main/TUM_associated.zip?download=true
unzip datasets/TUM_fr1.zip -d datasets/TUM_fr1
rm datasets/TUM_fr1.zip

mkdir results