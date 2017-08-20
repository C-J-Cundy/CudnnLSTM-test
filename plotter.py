#Adapted from Tom Runia, updated as EventAccumulator has been moved to tensorboard
import sys
import numpy as np
import seaborn as sns
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(log_files):
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", len(log_files))
    window = 20
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 1000,
        'histograms': 1
    }
#    labels = ['8192', '2048']

    for i, f in enumerate(log_files):
        print i,f
        event_acc = EventAccumulator(f, tf_size_guidance)
        event_acc.Reload()
        
        # Show all tags in the log file
        #print(event_acc.Tags())
        
        training_accuracies   = event_acc.Scalars('acc')        
        steps = np.shape(training_accuracies)[0]
        wall_zero = training_accuracies[0][0] #Wall-clock time @ step0
        x = np.zeros((steps), dtype='float32')
        y = np.zeros((steps), dtype='float32')
        
        for j in xrange(steps):
            y[j] = training_accuracies[j][2]
            x[j] = (training_accuracies[j][0] - wall_zero) / 60
            
        stds = pd.rolling_std(y, window)
        means = pd.rolling_mean(y, window)
            
            #Plot the thing
        ax.plot(x, means, c=clrs[i], label=f)
        ax.fill_between(x, means-stds, means+stds, alpha=0.3, facecolor=clrs[i])
    plt.xlabel("Wall Time / mins")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()
            
if __name__ == '__main__':
    log_files = sys.argv[1:]
    print log_files
    plot_tensorflow_log(log_files)
