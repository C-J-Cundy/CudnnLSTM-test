import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import math

#We construct an RNN to solve problem 2b in the original LSTM paper
#(Hochreiter & Schmidhuber 1997). We implement the LSTM layer using
#the BasicLSTM kernel provided by tensorflow. This is used as a baseline
#to compare against the CUDNN implementation

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

#Network Parameters
n_steps = 8192
n_hidden = 256
n_input = 257
n_classes = 2
n_layers = 1
sn = 1/math.sqrt(n_input) #Glorot initialisation

#Training Parameters
learning_rate = 0.002
training_iters = 5000000
batch_size = 128
display_step = 30

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

#Generate network
################################################################################

def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, num=n_steps, axis=0)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=5.0)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for _ in range(n_layers)])
    # Get lstm cell output
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('acc', accuracy)
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.device("gpu:0"):
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        test_writer = tf.summary.FileWriter('./BasicLSTM_'+str(n_steps)+'_stepslog', sess.graph)
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
                summary, _ = sess.run([merged, cost], feed_dict={x: batch_x, y: batch_y})
                summary, _ = sess.run([merged, accuracy], feed_dict={x: batch_x, y: batch_y})
                test_writer.add_summary(summary, step)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                step += 1
                if step % (display_step*10) == 0: #Save the model every so often
                    saver.save(sess, 'BasicLSTM_'+str(n_steps)+'_steps', global_step=step)
        print("Optimization Finished!")
