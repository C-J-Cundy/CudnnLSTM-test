#bin/bash
#This script fetches the necessary 

sudo apt-get --assume-yes install emacs
git clone https://github.com/C-J-Cundy/CudnnLSTM-test.git
git clone https://github.com/C-J-Cundy/.dotfiles.git
sudo mv ./.dotfiles/.emacs.d ~
sudo mv ./.dotfiles/.emacs ~
sudo mv ./.dotfiles/bash_profile ~
https://github.com/eamartin/plr.git
cd plr/linear_recurrent_net
sudo cp /usr/local/cuda/lib64/libcudart* /usr/lib/
./build.sh
export PYTHONPATH=~/plr/linear_recurrent_net:$PYTHONPATH

