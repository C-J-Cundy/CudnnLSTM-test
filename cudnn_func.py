import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import math
import time
#We construct an RNN to solve problem 2b in the original LSTM paper
#(Hochreiter & Schmidhuber 1997). We implement the LSTM layer using
#the CUDA kernel provided by tensorflow. There is little available
#documentation on using the CudnnLSTM bindings, so this provides a minimal
#working example for future reference.

#As far as I can tell this is the first nontrivial example of the CudnnLSTM
#bindings being used on the web. 

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


def cudnn(n_steps=1024, n_hidden=1024, n_input=128,
          batch_size=8, n_layers=1, n_converge=5, clip=20, sn=0.1, fg=5.0):
    #Network Parameters
    tf.reset_default_graph()
    n_classes = 2
    forget_gate_init = fg                          # = 1/(n_in). We use uniform p(x)

    #Training Parameters
    learning_rate = 0.0001
    training_iters = 5000000
    display_step = 10
    id_num = np.random.uniform(0, 50) #To distinguish from other runs of identical models


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
    with tf.device("gpu:0"):
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            test_writer = tf.summary.FileWriter('./CudnnLSTM_'+str(n_steps)+'_stepslog_' + str(id_num),
                                                sess.graph)
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
                        saver.save(sess, './CudnnLSTM_'+str(n_steps)+'_steps_model_' + str(id_num),
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
