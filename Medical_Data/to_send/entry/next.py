import tensorflow as tf
import numpy as np
import scipy.io.wavfile
from layers_new_cpu import linear_surrogate_lstm
import time
import os
import random
import sys
import scipy.signal
import warnings
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_wav(wav_path, downsample, n_steps, u=-489.3, sd=3759.16):
    """Loads a single ECG from a .wav file in order to test it"""
    data = scipy.io.wavfile.read(wav_path)[1]
    data = scipy.signal.decimate(data, downsample)        
    out = np.zeros((1, n_steps))
    out[0, n_steps - np.shape(data)[0]:] = (data - u) / sd
    return out 

n_hidden = 16
n_classes = 2
n_steps = 61000
batch_size = 1
n_input = 1
n_layers = 3
clip = 20 #We use gradient clipping to stop the gradient exploding initially
#for the much larger networks
downsample = 4

start = time.time()

#Initialise the model and evaluate
with tf.device("cpu:0"):
    with tf.Session() as sess:
        #Restore the graph
        new_saver = tf.train.import_meta_graph('lr_graph.meta')
        new_saver.restore(sess, 'lr_graph')
        pred = tf.get_collection('train_op')[0]
        x = tf.get_collection('x')[0]        
        x_in = load_wav(sys.argv[1] + '.wav', downsample, n_steps)[0].reshape((61000,1,1))
        out = sess.run(pred, feed_dict={x:x_in})
        with open('answers.txt', 'a') as g:
            if out[0][0] > out[0][1]:                            
                g.write(sys.argv[1] + ',' + '-1\n')
            else:
                g.write(sys.argv[1] + ',' + '1\n')
