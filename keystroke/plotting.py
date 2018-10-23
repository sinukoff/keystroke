import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keystroke import data_prep, features
from statsmodels.robust.scale import mad as MAD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
import pdb


def hist_flight_times(df=None):

    if df is None:
        df = data_prep.read_cleaned()

    bins = np.arange(0, 2000, 50)
    groups = df.groupby(['user_num', 'scenario_num']).groups.keys()

    grouplist = []
    pdb.set_trace()
    for i in groups:
        grouplist.append(
            df.loc[df.groupby(['user_num', 'scenario_num']).groups[i]]
        )

    diffs = np.array([])

    for dfi in grouplist:
        diffs = np.append(diffs, dfi.tstart.diff())
    pdb.set_trace()
    diffs = diffs[np.isnan(diffs) == False]
    diffs = pd.Series(diffs)
    plt.close('all')
    plt.hist(diffs, bins=bins)
    plt.xlabel('Flight Time (ms)')
    plt.ylabel('N')

    plt.savefig('/Users/evan/Code/Insight/plots/hist_flight_times.pdf', bbox_inches='tight', pad_inches=0.01)

    plt.close('all')



    groups = df.groupby(['user_num']).groups.keys()

    grouplist = []
    for i in groups:
        grouplist.append(df.loc[df.groupby(['user_num']).groups[i]])


    diffs = np.array([])

    for dfi in grouplist:
        diffs = dfi.tstart.diff()
        diffs = diffs[np.isnan(diffs) == False]
        diffs = pd.Series(diffs)

        plt.hist(diffs, bins=bins, histtype='step', lw=0.1, normed=True)
        plt.xlabel('Flight Time (ms)')
        plt.ylabel('N')

    plt.savefig('/Users/evan/Code/Insight/plots/hist_flight_times_users.pdf', bbox_inches='tight', pad_inches=0.01)

    plt.close('all')



def mean_std_per_user(df=None):

    if df is None:
        df = data_prep.read_cleaned()

    plt.close('all')

    groups = df.groupby(['user_num', 'scenario_num']).groups.keys()
    grouplist = []
    users = []
    for i in groups:
        grouplist.append(df.loc[df.groupby(['user_num', 'scenario_num']).groups[i]])
        users.append(i[0])

    for i, dfi in enumerate(grouplist):
        user = users[i]
        diffs = dfi.tstart.diff()
        diffs = diffs[np.isnan(diffs) == False]
        diffs = pd.Series(diffs)
        if len(diffs) < 100:
            continue

        #diffs = diffs[diffs < 1000.]
        med = np.median(diffs)
        std = np.std(diffs)
       # plt.errorbar(user, med, std, fmt='o', color='b')
        mad = MAD(diffs)
        plt.errorbar(user, med, mad, fmt='o', color='r')


    plt.savefig('/Users/evan/Code/Insight/plots/hist_flight_times_mean_MAD_users.pdf', bbox_inches='tight', pad_inches=0.01)

    plt.close('all')



def hist_scenarios_per_user(df=None):

    if df is None:
        df = features.read_features()

    plt.close('all')

    nscenarios = df.groupby(['user_num']).size()
    print(len(nscenarios[nscenarios > 3]))
    plt.hist(nscenarios, histtype='step', lw=1.0)
    plt.xlabel('N Scenarios')
    plt.ylabel('N Users')
    plt.savefig('/Users/evan/Code/Insight/plots/hist_scenarios_per_user.pdf', bbox_inches='tight', pad_inches=0.01)

    plt.close('all')
    return



def hist_keys_per_scenario(df=None):

    if df is None:
        df = data_prep.read_cleaned()

    plt.close('all')

    nscenarios = df.groupby(['user_num', 'scenario_num']).size()
    #print(len(nscenarios[nscenarios > 3]))
    bins=np.arange(0, 4000, 100)
    plt.hist(nscenarios, bins=bins, histtype='step', lw=1.0)
    plt.xlabel('N Keys')
    plt.ylabel('N Scenarios')
    plt.savefig('/Users/evan/Code/Insight/plots/hist_keys_per_scenario.pdf', bbox_inches='tight', pad_inches=0.01)


def plot_tsne(X, y, nchunks, perplexity, learnrate, saveto):
    from scipy.stats import rankdata as rd
    #users = df.user_num.values
    plt.close('all')
   # X_embedded = TSNE(learning_rate=20, perplexity=4, n_iter=100000, n_iter_without_progress=1000, init='random').fit(X)
   # print(X_embedded.kl_divergence_)
    X_embedded = TSNE(perplexity=perplexity, learning_rate=learnrate, n_iter=100000, n_iter_without_progress=1000, init='random', random_state=25).fit_transform(X)
    print(silhouette_score(X_embedded, y))
    xs = X_embedded[:, 0]
    ys = X_embedded[:, 1]
    xmin = np.argmin(xs)
    xmax = np.argmax(xs)
    ymin = np.argmax(ys)
    ymax = np.argmax(ys)

    # xs = np.delete(xs, [xmin, xmax, ymin, ymax])
    # ys = np.delete(ys, [xmin, xmax, ymin, ymax])
    # users = np.delete(users, [xmin, xmax, ymin, ymax])
    users = np.unique(y)
    for i in range(int(len(users))):
        y[y == users[i]] = i
   # for i in range(int(len(xs)/nchunks)):
   #     user = y[i*nchunks]
   #     y[i*nchunks:(i+1)*nchunks] = i
   # plt.scatter(xs, ys, c=y, s=25, alpha=1, cmap='tab20')
    plt.scatter(xs, ys, c=y, s=100, cmap='tab20b', alpha=1.0, edgecolor='None')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('t-SNE Dimension 1', fontsize=13)
    plt.ylabel('t-SNE Dimension 2', fontsize=13)
   # plt.scatter(xs, ys, c=y, s=25, alpha=1, cmap='tab20c')
   # plt.scatter(xs, ys, c=y, s=25, alpha=1, cmap='gist_ncar')
    cmap = matplotlib.cm.get_cmap('tab20b')
   # pdb.set_trace()
    for i in range(len(xs)):
         marker='${}$'.format(y[i])
       #  print(y[i]/np.max(y))
    #    # pdb.set_trace()
         color = cmap(y[i]/np.max(y))
         if i < 10:
             size = 25
         else:
             size = 50
        # print(color)
         #plt.scatter(xs[i], ys[i], c=color, s=size, marker=marker, alpha=1, cmap='tab20b')
    #for i in range(len(xs)):
       #  plt.text(xs[i], ys[i], str(y[i]), color='k', ha='center', va='center', fontsize=10)# + ' ' + str(scenarios[i]))
   # pdb.set_trace()
    plt.savefig(saveto, bbox_inches='tight', pad_inches=0.01)
    plt.close('all')

    return


def plot_roc(trueneg, truepos):

    plt.close('all')
    plt.plot(1.0 - np.array(trueneg), truepos, color='b')
    plt.xlabel('False Positives', fontsize=14)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('Fraction of Imposters Identified', fontsize=14)
    plt.plot([0, 1], [0, 1], '--', color='k', lw=0.5)
    ax=plt.gca()
    plt.grid(lw=0.3)
    plt.savefig('/Users/evan/roctest.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')

    #plt.