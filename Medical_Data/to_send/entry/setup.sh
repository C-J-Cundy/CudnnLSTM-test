#bin/bash
#This script sets up the model, sets the tf model looking for input
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared LinearRecurrence2.cc -o LinearRecurrenceNew.so -fPIC -I $TF_INC -O2
