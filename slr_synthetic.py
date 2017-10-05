import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import math
from layers_new import linear_surrogate_lstm
import time
import os

#We use an experimental parallel linear recurrence architecture for computing
#the passes

#We wish to train a network to classify sequences of random vectors on the unit
#cube, where the first vector's sign determines the (binary) classification of
#the sequence. To save space, we limit the length of the alphabet to 128 (in the
#original paper the length of the sequence was the length of the alphabet, which
#makes the space requirements grow very rapidly).


#Generate training data batch
def gen_2a_data(p, bs):
    x_out = np.zeros((p+1, bs, p+1))
    y_out = np.zeros((bs,2)).zeros((bs,2))
    for i in range(bs/2):
        x_out[:,i,:] = np.eye(p+1)
        x_out[0,i,0] = 1
        y_out[i, 0] = 1
    for i in range(bs/2, bs):
        x_out[:,i,:] = np.eye(p+1)
        x_out[0,i,0] = -1
        y_out[i, 1] = 1
    return x_out, y_out

def gen_2b_data(p, q, bs):
    x_out = np.zeros((p+1, bs, q+1))
    y_out = np.zeros((bs,2))
    for i in range(bs/2):
        x_out[:,i,:] = np.eye(q+1)[np.random.choice(q+1, p+1)] #Random one-hot
        x_out[0,i,:] = np.zeros(q+1) 
        x_out[0,i,0] = 1 #Set indicator component
        y_out[i, 0] = 1 
    for i in range(bs/2, bs):        
        x_out[:,i,:] = np.eye(q+1)[np.random.choice(q+1, p+1)]
        x_out[0,i,:] = np.zeros(q+1)
        x_out[0,i,0] = -1
        y_out[i, 1] = 1
    perm = np.random.permutation(bs) #Shuffle order of outputs
    return x_out[:, perm, :], y_out[perm]


def gen_2b_data_1(p, q):
    """Generates data with a batch size of 1"""
    x_out = np.zeros((p+1, 1, q+1))
    y_out = np.zeros((1,2))
    seq_class = np.random.random_integers(2) - 1
    if seq_class == 0:
        x_out[:,0,:] = np.eye(q+1)[np.random.choice(q+1, p+1)] #Random one-hot
        x_out[0,0,:] = np.zeros(q+1) 
        x_out[0,0,0] = 1 #Set indicator component
        y_out[0, 0] = 1 
    else:
        x_out[:,0,:] = np.eye(q+1)[np.random.choice(q+1, p+1)]
        x_out[0,0,:] = np.zeros(q+1)
        x_out[0,0,0] = -1
        y_out[0, 1] = 1
    return x_out, y_out


def slr(n_steps=1024, n_hidden=1024, n_input=128, batch_size=8, n_layers=1, n_converge=5):
        #Network Parameters
    tf.reset_default_graph()
    n_classes = 2
    sn = 1/math.sqrt(n_hidden) #Glorot initialisation, var(p(x))
    forget_gate_init = 5.0                          # = 1/(n_in). We use uniform p(x)
    clip = 20 #We use gradient clipping to stop the gradient exploding initially
    #for the much larger networks


    #Training Parameters
    learning_rate = 0.0001
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
    layer1 = SRU(x, name='SRU')
    outputs = SRU(layer1, name='SRU2')    

    # Linear activation, using rnn inner loop last output
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
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    start = time.time()
    acc_list = [0]*n_converge
    if not os.path.exists('./LS_LSTM_'+str(n_steps)+'_steps_model_'):
        os.makedirs('./LS_LSTM_'+str(n_steps)+'_steps_model_')
    if not os.path.exists('./LS_LSTM_'+str(n_steps)+'_steps_log_'):
        os.makedirs('./LS_LSTM_'+str(n_steps)+'_steps_log_')
    
    with tf.device("gpu:0"):
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            test_writer = tf.summary.FileWriter('./LS_LSTM_'+str(n_steps)+'_stepslog_', sess.graph)
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                if batch_size == 1:
                    batch_x, batch_y = gen_2b_data_1(n_steps-1, n_input-1)
                else:
                    batch_x, batch_y = gen_2b_data(n_steps-1, n_input-1, batch_size)
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
                          "{:.5f}".format(acc))
                    if step % (display_step*10) == 0: #Save the model every so often
                        saver.save(sess, './LS_LSTM_'+str(n_steps)+'_steps_model_',
                                   global_step=step)
                    if acc_list == [1.0]*n_converge:
                        print "Converged after {} iterations and {} seconds".format(step, time.time() - start)
                        break
                    else:
                        acc_list.append(acc)
                        acc_list.pop(0)                    
                step += 1

            print("Optimization Finished!")
    return step, time.time()-start


if __name__ == '__main__':
    slr()
