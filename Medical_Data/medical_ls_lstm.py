import tensorflow as tf
import numpy as np
import scipy.io.wavfile
from tensorflow.contrib import rnn
import math
from layers_new import linear_surrogate_lstm
import time
import os
import random


def load_wavs(labels, prefix):
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
        out_x[index, length - np.shape(data)[0]:] = data
        if ground_truth[f[:-4]] == 1:
            out_y[index, 1] = 1
        else:
            out_y[index, 0] = 1                
    return out_x, out_y
            

labels = ['a', 'b', 'c', 'd', 'e', 'f']
prefix = 'training-'
xs, ys = load_wavs(labels, prefix)
val_xs, val_ys = load_validation(np.shape(xs)[1])
print np.sum(val_ys, axis=1)

n_hidden = 256
n_classes = 2
n_steps = np.shape(xs)[1]
batch_size = 1
n_input = 1
n_layers = 2
sn = 1/math.sqrt(n_hidden) #Glorot initialisation, var(p(x))
forget_gate_init = 1.0                          # = 1/(n_in). We use uniform p(x)
clip = 20 #We use gradient clipping to stop the gradient exploding initially
#for the much larger networks


#Training Parameters
learning_rate = 0.001
training_iters = 5000000
display_step = 10
id_num = np.random.uniform(0, 50) #To distinguish from other runs of identical models


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

layer1 = linear_surrogate_lstm(x, n_hidden, name='ls-lstm')
outputs = linear_surrogate_lstm(layer1, n_hidden, name='ls-lstm2')    
pred = tf.matmul(outputs[-1], W1) + b1

#Evaluate network, run adam and clip gradients
################################################################################
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
gradients, _ = tf.clip_by_global_norm(raw_gradients, clip)
optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('cost', cost)
tf.summary.scalar('acc', accuracy)
tf.summary.scalar('val_acc', accuracy)
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()

start = time.time()
if not os.path.exists('./LS_LSTM_'+str(n_steps)+'_steps_model_'):
    os.makedirs('./LS_LSTM_'+str(n_steps)+'_steps_model_')
if not os.path.exists('./LS_LSTM_'+str(n_steps)+'_steps_log_'):
    os.makedirs('./LS_LSTM_'+str(n_steps)+'_steps_log_')

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

xs_chunked = list(chunker(xs, batch_size))[:-1]
ys_chunked = list(chunker(ys, batch_size))[:-1]
acc_list = []
n_converge = 30

with tf.device("gpu:0"):
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        epoch_counter = 0
        test_writer = tf.summary.FileWriter('./LS_LSTM_'+str(n_steps)+'_stepslog_', sess.graph)
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = np.reshape(xs_chunked[step % (int(math.floor(float(np.shape(xs)[0]) / batch_size) - 1))], (np.shape(xs)[1], batch_size, 1)), ys_chunked[step % (int(math.floor(float(np.shape(xs)[0]) / batch_size) - 1))]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                summary, _ = sess.run([merged, cost], feed_dict={x: batch_x, y: batch_y})
                summary, _ = sess.run([merged, accuracy], feed_dict={x: batch_x, y: batch_y})
                test_writer.add_summary(summary, step)
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc) + ", Validation Accuracy= ")
                

                if step % (display_step*10) == 0: #Save the model every so often
                    saver.save(sess, './LS_LSTM_'+str(n_steps)+'_steps_model_/',
                               global_step=step)
                    val_accs = []
                    for index in range(len(val_xs)):
                        val_accs.append(sess.run(accuracy, feed_dict={x: np.reshape(val_xs[index], (np.shape(val_xs)[1], 1, 1)), y: np.reshape(val_ys[index], (1, 2))}))
                    val_acc = np.mean(val_accs)
                    print("{:.5f}".format(val_acc))
                    test_writer.add_summary(val_acc, step)                              
                if acc_list == [1.0]*n_converge:
                    print "Converged after {} iterations and {} seconds".format(step, time.time() - start)
                    break
                else:
                    acc_list.append(acc)
                    acc_list.pop(0)                    
            step += 1
            if step % (math.floor(float(np.shape(xs)[0]) / batch_size) - 1) == 0:
                print "Epoch {} finished".format(epoch_counter)
                epoch_counter += 1
                #Shuffle dataset
                dual_list = list(zip(xs_chunked, ys_chunked))
                random.shuffle(dual_list)
                xs_chunked, ys_chunked = zip(*dual_list)
        print("Optimization Finished!")




