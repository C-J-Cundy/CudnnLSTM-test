import numpy as np
from cudnn_func import cudnn
from ls_lstm_func import ls_lstm

num_trials = 5
n_steps = [1024, 8192, 1048576]

for n_step in n_steps:
    #Do the iteration for the  2-layer cudnn
    #Make sure that the sn is set to 0.03 for 8192 for the cuda
    n_hidden = 1024
    iter_list = []
    times_list = []
    for _ in range(num_trials):
        a, b = cudnn(n_step, n_hidden, 128, 16, 1)
        iter_list.append(a)
        times_list.append(b)
        print "Took {} seconds to converge after {} iterations".format(b, a)        
    print """After {} trials for the one-layer cudnn, with sequence length
    {}, n_hidden {}, on average took {} pm {} iterations and {} pm {} seconds
    to converge""".format(num_trials, n_step, n_hidden, np.mean(iter_list),
                          np.std(iter_list), np.mean(times_list), np.std(times_list))


    n_hidden = 512
    iter_list = []
    times_list = []
    for _ in range(num_trials):
        a, b = cudnn(n_step, n_hidden, 128, 16, 2)
        iter_list.append(a)
        times_list.append(b)
        print "Took {} seconds to converge after {} iterations".format(b, a)
    print """After {} trials for the 2-layer cudnn, with sequence length
    {}, n_hidden {}, on average took {} pm {} iterations and {} pm {} seconds
    to converge""".format(num_trials, n_step, n_hidden, np.mean(iter_list),
                          np.std(iter_list), np.mean(times_list), np.std(times_list))

    
    n_hidden = 512
    iter_list = []
    times_list = []
    for _ in range(num_trials):
        a, b = ls_lstm(n_step, n_hidden, 128, 8, 2)
        iter_list.append(a)
        times_list.append(b)
    print """After {} trials for the one-layer cudnn, with sequence length
    {}, n_hidden {}, on average took {} pm {} iterations and {} pm {} seconds
    to converge""".format(num_trials, n_step, n_hidden, np.mean(iter_list),
                          np.std(iter_list), np.mean(times_list), np.std(times_list))      
