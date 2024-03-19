# usage: source install.sh

conda create -n SP python=3.8 -y
conda activate SP
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install open3d==0.17.0 cupy-cuda11x==12.1.0 pytorch-pfn-extras==0.7.6 opencv-python==4.7.0.72 tqdm pathlib trimesh geffnet==1.0.2 prettytable --no-input
pip install evo --upgrade --no-binary evo --no-input

# install lietorch 
git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch
python setup.py install
pip install -e . 
cd .. 

# install segment anything
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..