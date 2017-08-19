import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import math

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

#Training Parameters
learning_rate = 0.005
training_iters = 500000
batch_size = 1024
display_step = 10

#Network Parameters
n_steps = 20148
n_hidden = 127
n_input = 128
n_classes = 2
n_layers = 1
sn = 1/math.sqrt(n_input) #Glorot initialisation

#Initialise variables
################################################################################

#Generate dummy model so that we can use its method later on
model = tf.contrib.cudnn_rnn.CudnnLSTM(n_layers, n_hidden, n_input)

# tf Graph input
x = tf.placeholder("float", [n_steps, batch_size, n_input])
y = tf.placeholder("float", [batch_size, n_classes])

# Define weights & parameters
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
#There are eight weights and biases per layer in the LSTM, 
weight_list = []
bias_list = []
for n in range(4):
    weight_list.append(np.float32(
        np.random.uniform(low=-sn, high=sn,
                          size=[n_input, n_hidden])))

for n in range(4,8):
    weight_list.append(np.float32(
        np.random.uniform(low=-sn, high=sn,
                          size=[n_hidden, n_hidden])))

for n in range(8):
    bias_list.append(np.float32(
        np.zeros([n_hidden])))

#Set forget gate bias high to ensure no early forgetting
bias_list[1] = np.float32(
        20*np.ones([n_hidden]))

bias_list[5] = np.float32(
        20*np.ones([n_hidden]))

params = {
    'out': model.canonical_to_params(weight_list, bias_list)
}

#Generate network
################################################################################

def RNN(x, weights, biases, input_h, input_c, params):
    rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(n_layers, n_hidden, n_input)
    outputs, states1, states2 = rnn_cell(
        is_training=True,
        input_data=x,
        input_h=input_h['out'],
        input_c=input_c['out'],
        params=params['out'])
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases, input_h, input_c, params)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
acc = tf.summary.scalar('acc', accuracy)
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    train_writer = tf.summary.FileWriter('./', sess.graph)
    test_writer = tf.summary.FileWriter('./', sess.graph)
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = gen_2b_data(n_steps-1, n_input-1, batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            summary, acc = sess.run([merged, cost], feed_dict={x: batch_x, y: batch_y})
            test_writer.add_summary(summary, step)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
