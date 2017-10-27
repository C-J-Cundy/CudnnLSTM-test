def random_test(bs_seq_len_list):
    """Given a list of pairs (batch size, seq_len), 
    calculate the throughput of an LS-LSTM vs a cudnn on 
    random data"""
    import tensorflow as tf
    import numpy as np
    import scipy.io.wavfile
    from tensorflow.contrib import rnn
    import math
    from layers_new import linear_surrogate_lstm
    import time
    import os
    import random

    ls_lstm_throughput_dict = {}
    cudnn_throughput_dict = {}
    for bs, seq_len in bs_seq_len_list:
        #First generate the LS-LSTM and work out the throughput
        tf.reset_default_graph()        
        n_hidden = 234
        n_classes = 2
        n_steps = seq_len
        batch_size = bs
        n_input = 4
        n_layers = 2
        forget_gate_init = 1.0                          # = 1/(n_in). We use uniform p(x)
        sn = 1.0 / math.sqrt(n_hidden)
        #Training Parameters
        learning_rate = 0.001
        training_iters = 5000000

        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_hidden, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')

        layer1 = linear_surrogate_lstm(x, n_hidden, name='ls-lstm')
        outputs = linear_surrogate_lstm(layer1, n_hidden, name='ls-lstm2')    
        pred = tf.matmul(outputs[-1], W1) + b1

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        ls_lstm_throughput_dict[(bs, n_steps)] = (bs * n_steps) / np.mean(times)

        #--------------------------------------------------------------------------------
        # Now we do the CUDNN    
        tf.reset_default_graph()

        #Initialise variables
        ################################################################################
        #Generate the lstm hook to CUDA
        model = tf.contrib.cudnn_rnn.CudnnLSTM(n_layers, n_hidden, n_input)

        # tf Graph input
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])

        #Define weights & rnn initial states
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), dtype='float')
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]), dtype='float')
        }
        #Initial state of the LSTM at each batch, we don't let this be trained. 
        input_h = {
            'out': tf.Variable(tf.zeros([n_layers, batch_size, n_hidden]), dtype='float',
                               trainable=False)
        }
        input_c = {
            'out': tf.Variable(tf.zeros([n_layers, batch_size, n_hidden]), dtype='float',
                               trainable=False)
        }
        #Initialise all weights & biases for the cudnnlstm: set weights according to Glorot
        #There are eight weights and biases per layer in the LSTM. Described in 
        #http://docs.nvidia.com/deeplearning/sdk/cudnn-user-guide/index.html#cudnnRNNMode_t
        #There are two biases which sum to give the biases in the canonical form of the LSTM
        #This seems redundant - I'm not sure why CUDA is implemented in this way.

        weight_list = []
        bias_list = []
        for n in range(4):
            weight_list.append(np.float32(
                np.random.uniform(low=-sn, high=sn,
                                  size=[n_hidden, n_input])))

        for n in range(4,8):
            weight_list.append(np.float32(
                np.random.uniform(low=-sn, high=sn,
                                  size=[n_hidden, n_hidden])))
        if n_layers == 2:
            for n in range(4):
                weight_list.append(np.float32(
                    np.random.uniform(low=-sn, high=sn,
                                      size=[n_hidden, n_hidden])))

            for n in range(4,8):
                weight_list.append(np.float32(
                    np.random.uniform(low=-sn, high=sn,
                                      size=[n_hidden, n_hidden])))        

        for n in range(8):
            bias_list.append(np.float32(
                np.zeros([n_hidden])))
        if n_layers == 2:
            for n in range(8):        
                bias_list.append(np.float32(
                    np.zeros([n_hidden])))
            bias_list[13] = np.float32(
                forget_gate_init*np.ones([n_hidden]))

        bias_list[5] = np.float32(
                forget_gate_init*np.ones([n_hidden]))

        #Initialize the opaque parameter buffer used to handle the cudnnlstm params
        #If we try to pass the canonical_to_params tensor through the call graph,
        #we fail because the size must be known statically. The easiest way to get
        #around this (though hacky) is to get the values out by casting to an np array
        #and then initialising a tensor with those values.

        params_size_t = (  (n_input * n_hidden * 4) 
                         + (n_hidden * n_hidden * 4)
                         + (n_hidden * 2 * 4))
        flat_params = model.canonical_to_params(weight_list, bias_list)
        flat_params_as_ndarray = tf.Session().run(flat_params) 

        params = {
            'out': tf.get_variable('param_buffer', initializer=tf.constant(
                flat_params_as_ndarray))
            }


        #Generate network
        ################################################################################
        outputs, states1, states2 = model(
            is_training=True,
            input_data=x,
            input_h=input_h['out'],
            input_c=input_c['out'],
            params=params['out'])

        # Linear activation, using rnn inner loop on last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']    

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        step = 0
        times = []
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        cudnn_throughput_dict[(bs, n_steps)] = (bs * n_steps) / np.mean(times)
    return cudnn_throughput_dict, ls_lstm_throughput_dict


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def to_df(d):
    return pd.DataFrame([[k[0], k[1], v] for k, v in d.items()],
                        columns=['bs', 'seqlen', 'val'])

def plot(fast_tp, slow_tp):
    # fast_tp and slow_tp are dicts from {(bs, seqlen): throughput} with throughput in steps/s
    # it's ok if values are missing in the dicts. See Seaborn heatmap docs.
    # for plot in pre-print, slow_tp is LSTM throughput and fast is LS-LSTM
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    def to_df(d):
        return pd.DataFrame([[k[0], k[1], v] for k, v in d.items()],
                            columns=['bs', 'seqlen', 'val'])


    # plot throughput values in tp_scale steps/s
    tp_scale = 1000.

    max_tp = max(max(fast_tp.values()), max(slow_tp.values())) / tp_scale

    props = {'height_ratios': (.1, .9)}
    f, ax = plt.subplots(2, gridspec_kw=props)
    plt.subplots_adjust(left=.2, right=.85)

    df = to_df(slow_tp)
    df = df[df.seqlen == 512]    # only plot 1 row of slow throughput (because LSTM throughput doesn't depend on seqlen)
    df = df.pivot('seqlen', 'bs', 'val') / tp_scale
    sns.heatmap(df, ax=ax[0], annot=True, fmt='.3g', cbar=False, yticklabels=[''],
                vmin=0, vmax=max_tp)
    ax[0].yaxis.label.set_visible(False)
    ax[0].set_xlabel('batch size')
    ax[0].xaxis.set_label_coords(-.18, -.25)
    ax[0].annotate('LSTM', xy=(1, 0), xytext=(1.03, .4), textcoords='axes fraction',
                   fontsize=14)

    df = to_df(fast_tp).pivot('seqlen', 'bs', 'seqlen') / tp_scale
    sns.heatmap(df, ax=ax[1], annot=True, fmt='.3g', cbar=False, xticklabels=[''],
                vmin=0, vmax=max_tp)
    ax[1].set_ylabel('seq length', rotation=0)
    ax[1].yaxis.set_label_coords(-.18, .485)
    plt.setp(ax[1].yaxis.get_majorticklabels(), rotation=0)
    ax[1].set_xlabel('')
    ax[1].annotate('LS-LSTM', xy=(1, 0), xytext=(1.03, .485), textcoords='axes fraction',
                   fontsize=14)

    plt.show()

if __name__ == "__main__":
    import pickle
    fast_tp = {(bs, 65536 / bs): 5 * bs for bs in range(1, 8)}
    slow_tp = {(bs, 32768 / bs): 1 * bs for bs in range(2, 8)}
    in_list1 = [[1, x] for x in [2**z for z in range(8, 19-1)]]
    in_list2 = [[2, x] for x in [2**z for z in range(8, 19-2)]]
    in_list4 = [[4, x] for x in [2**z for z in range(8, 19-3)]]
    in_list8 = [[8, x] for x in [2**z for z in range(8, 19-4)]]
    in_list16 = [[16, x] for x in [2**z for z in range(8, 19-5)]]
    in_list32 = [[32, x] for x in [2**z for z in range(8, 19-6)]]
    in_list64 = [[64, x] for x in [2**z for z in range(8, 19-7)]]
    in_list128 = [[128, x] for x in [2**z for z in range(8, 19-8)]]
    in_list256 = [[256, x] for x in [2**z for z in range(8, 19-9)]]                                

    in_list1.extend(in_list2)
    in_list1.extend(in_list4)
    in_list1.extend(in_list8)
    in_list1.extend(in_list16)
    in_list1.extend(in_list32)
    in_list1.extend(in_list64)
    in_list1.extend(in_list128)
    in_list1.extend(in_list256)

    out = random_test(in_list1)
    print out
    lstm_times, cudnn_times = out

    fileObject = open('./lstm_dict', 'wb')
    pickle.dump(lstm_times, fileObject)
    fileObject.close()

    fileObject = open('./cudnn_dict', 'wb')
    pickle.dump(cudnn_times, fileObject)
    fileObject.close()
