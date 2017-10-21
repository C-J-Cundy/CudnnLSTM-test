#bin/bash
#This script fetches the necessary dependencies and editor configs
sudo add-apt-repository ppa:kelleyk/emacs
sudo apt-get update
sudo apt-get --assume-yes install emacs25
git clone https://github.com/C-J-Cundy/CudnnLSTM-test.git
git clone https://github.com/C-J-Cundy/.dotfiles.git
sudo mv ./.dotfiles/.emacs.d ~
sudo mv ./.dotfiles/.emacs ~
sudo mv ./.dotfiles/.bash_profile ~
git clone https://github.com/eamartin/plr.git
cd plr/linear_recurrent_net
sudo cp /usr/local/cuda/lib64/libcudart* /usr/lib/
./build.sh
export PYTHONPATH=~/plr/linear_recurrent_serial:~/plr/linear_recurrent_net:$PYTHONPATH
alias emacs=emacs25
