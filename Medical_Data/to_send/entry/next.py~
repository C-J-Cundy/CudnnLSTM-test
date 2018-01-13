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

def load_wav(wav_path, downsample, n_steps):
    """Loads a single ECG from a .wav file in order to test it"""
    data = scipy.io.wavfile.read(wav_path)[1]
    data = scipy.signal.decimate(data, downsample)        
    out = np.zeros((1, n_steps))
    out[0, n_steps - np.shape(data)[0]:] = data
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
        print 'Finished setting up'

        ref = time.time()
        cur_file = ''
        while True:
            time.sleep(0.1)
            #We wait for the in_file to be updated
            with open('in_list') as f:
                rl = f.readlines()
                if rl == []:
                    pass
                else:
                    if rl[0].strip('\n') != cur_file:
                        start = time.time()
                        wav_path = rl[0].strip('\n')
                        cur_file = wav_path                            
                        x_in = load_wav(wav_path, downsample, n_steps)[0]
                        print x_in                            
                        x_in = x_in.reshape((n_steps,1,1))
                        out = sess.run(pred, feed_dict={x: x_in})
                        with open('out_file', 'wb') as g:
                            if out[0][0] > out[0][1]:                            
                                g.write('-1')
                            else:
                                g.write('1')
                        print "Result was ", out
                        print "Done, took {}".format(time.time() - start)
