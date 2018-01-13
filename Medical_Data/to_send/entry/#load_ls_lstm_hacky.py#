def load_compute(params_path, wav_path):
    """Given the path to a saved mode and the path to a .wav file,
    load the graph into memory and compute the classification of the
    .wav file"""
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
#    tf.logging.set_verbosity(tf.logging.ERROR)
#    warnings.simplefilter(action='ignore', category=FutureWarning)

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


    #Training Parameters
    #Initialise variables
    ################################################################################
    #Generate the lstm hook to PLR

    # tf Graph input
    x = tf.placeholder("float", [n_steps, batch_size, n_input])
    y = tf.placeholder("float", [batch_size, n_classes])

    #Define weights & rnn initial states

    tf.get_variable_scope().reuse == True
    W1 = tf.get_variable('W1', initializer=
                         tf.random_normal([n_hidden, n_classes]), dtype='float')
    b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')

    

    #Initialise all weights & biases for the plrlstm: set weights according to Glorot
    #There are eight weights and 4 biases per layer in the LSTM. Described in
    #http://docs.nvidia.com/deeplearning/sdk/cudnn-user-guide/index.html#cudnnRNNMode_t
    #There are two biases which sum to give the biases in the canonical form of the LSTM

    #Generate network
    ################################################################################
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(file_name='./model.ckpt-399000', tensor_name='', all_tensors=False)

    layer1 = linear_surrogate_lstm(x, n_hidden, name='ls-lstm')
    layer2 = linear_surrogate_lstm(layer1, n_hidden, name='ls-lstm2')
    outputs = linear_surrogate_lstm(layer2, n_hidden, name='ls-lstm3')
    
    pred = tf.matmul(outputs[-1], W1) + b1

    #Evaluate network, run adam and clip gradients
    ################################################################################
    start = time.time()

 #   print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#    for pp in tf.get_collection('variables'):
#        print pp 
    #Initialise the model and evaluate
    downsample=4
    tf.add_to_collection('train_op', pred)
    tf.add_to_collection('x', x)    
    with tf.device("cpu:0"):
        with tf.Session() as sess:
            var_dict = {}
            for variable in tf.get_collection('variables'):
                var_dict[variable.name[:-2]] = variable
                print variable.name[:-2], variable                
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, params_path)
            print "Done setting up!"
            saver0 = tf.train.Saver()
            saver0.save(sess, 'lr_graph')            
            ref = time.time()

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
        
def load_wavs(labels, prefix, downsample=1):
    """Loads wav files from directories of the form [labels]prefix,
    and returns the array with the data in. The data is padded up to the
    longest length"""
    import sys
    max_len = 0
    num = 0    
    for label in labels:
        files = os.listdir(prefix + label)
        for f in files:
            if f[-3:] == 'wav':
                data = scipy.io.wavfile.read(prefix + label + '/' + f)[1]
                data = scipy.signal.decimate(data, downsample)
                if np.shape(data)[0] > max_len:
                    max_len = np.shape(data)[0]
                num += 1
    out_x = np.zeros((num, max_len))
    out_y = np.zeros((num,2))    
    x_pos = 0
    for label in labels:
        files = os.listdir(prefix + label)
        for f in files:
            if f[-3:] == 'wav':
                data = scipy.io.wavfile.read(prefix + label + '/' + f)[1]
                data = scipy.signal.decimate(data, downsample)                
                out_x[x_pos, max_len - np.shape(data)[0]:] = data
                with open(prefix + label + '/' + f[:-3] + 'hea') as g:
                    if g.readlines()[-1] == "# Abnormal\r\n":
                        out_y[x_pos, 1] = 1
                    else:
                        out_y[x_pos, 0] = 1                
                x_pos += 1
    return out_x, out_y

def load_validation(length):
    import sys
    from pandas import read_csv
    path = './validation'
    files = os.listdir(path)
    nums = len([f for f in files if f[-3:] == 'wav'])
    out_x = np.zeros((nums, length))
    out_y = np.zeros((nums, 2))    
    from numpy import genfromtxt
    ground_truth = dict(np.array(read_csv('./validation/REFERENCE.csv')))
    ground_truth['a0001'] = 1
    for index, f in enumerate([f for f in files if f[-3:] == 'wav']):
        data = scipy.io.wavfile.read(path + '/' + f)[1]
        data = scipy.signal.decimate(data, downsample)        
        out_x[index, length - np.shape(data)[0]:] = data
        if ground_truth[f[:-4]] == 1:
            out_y[index, 1] = 1
        else:
            out_y[index, 0] = 1                
    return out_x, out_y


if __name__ == '__main__':
    import sys
    load_compute(sys.argv[1], sys.argv[2])
