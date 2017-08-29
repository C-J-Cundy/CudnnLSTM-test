#bin/bash

sudo apt-get install emacs
git clone https://github.com/C-J-Cundy/CudnnLSTM-test.git
git clone https://github.com/C-J-Cundy/.dotfiles.git
mv ./dotfiles/.emacs.d ~
mv ./dotfiles/.emacs ~
mv ./dotfiles/bash_profile ~
git clone CudnnLSTM-test.git
