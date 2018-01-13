#bin/bash
#This script sets up the model, sets the tf model looking for input
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared LinearRecurrence2.cc -o LinearRecurrenceNew.so -fPIC -I $TF_INC -O2

touch 'in_list'
python load_ls_lstm_hacky.py './model.ckpt-399000' foo & 
sleep 15
