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
    print max_tp

    props = {'height_ratios': (.1, .9)}
    f, ax = plt.subplots(2, gridspec_kw=props)
    plt.subplots_adjust(left=.2, right=.85)

    df = to_df(slow_tp)
    df = df[df.seqlen == 512]    # only plot 1 row of slow throughput (because LSTM throughput doesn't depend on seqlen)
    df = df.pivot('seqlen', 'bs', 'val') / tp_scale
    sns.heatmap(df, ax=ax[0], annot=True, fmt='.3g', cbar=False, yticklabels=[''],
                vmin=0, vmax=max_tp, cmap=sns.cubehelix_palette(reverse=False))
    ax[0].yaxis.label.set_visible(False)
    ax[0].set_xlabel('batch size')
    ax[0].xaxis.set_label_coords(-.18, -.25)
    ax[0].annotate('LSTM', xy=(1, 0), xytext=(1.03, .4), textcoords='axes fraction',
                   fontsize=14)

    df = to_df(fast_tp).pivot('seqlen', 'bs') / tp_scale
    sns.heatmap(df, ax=ax[1], annot=True, fmt='.3g', cbar=False, xticklabels=[''],
                vmin=5, vmax=max_tp, cmap=sns.cubehelix_palette(reverse=False))
    ax[1].set_ylabel('seq length', rotation=0)
    ax[1].yaxis.set_label_coords(-.18, .485)
    plt.setp(ax[1].yaxis.get_majorticklabels(), rotation=0)
    ax[1].set_xlabel('')
    ax[1].annotate('LS-LSTM', xy=(1, 0), xytext=(1.03, .485), textcoords='axes fraction',
                   fontsize=14)

    plt.show()

if __name__ == "__main__":
    import pickle
    lstm_dict = pickle.load(open('./lstm_dict', 'r'))
    cudnn_dict = pickle.load(open('./cudnn_dict', 'r'))
    print 'lstm_dict is ', lstm_dict
    print  'cudnn dict is', cudnn_dict
    plot(lstm_dict, cudnn_dict)
