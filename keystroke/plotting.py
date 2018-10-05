import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keystroke import data_prep, features
from statsmodels.robust.scale import mad as MAD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score


import pdb

def hist_flight_times(df=None):

    if df is None:
        df = data_prep.read_cleaned()

    bins = np.arange(0, 2000, 50)
    groups = df.groupby(['user_num', 'scenario_num']).groups.keys()

    grouplist = []
    for i in groups:
        grouplist.append(df.loc[df.groupby(['user_num', 'scenario_num']).groups[i]])


    diffs = np.array([])

    for dfi in grouplist:

        diffs = np.append(diffs, dfi.tstart.diff())
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


def plot_2dhist():

    A = np.zeros([8, 8])
    A[1-1, 1-1] = -0.37
    A[1-1, 2-1] = -0.42
    A[1-1, 3-1] = -0.40
    A[1-1, 4-1] = -0.50
    A[1-1, 5-1] = -0.49
    A[1-1, 6-1] = -0.40
    A[1-1, 7-1] = -0.44
    A[1-1, 8-1] = -0.49
    A[2-1, 1-1] = -0.42
    A[2-1, 2-1] = -0.42
    A[2-1, 3-1] = -0.26
    A[2-1, 4-1] = -0.37
    A[2-1, 5-1] = -0.38
    A[2-1, 6-1] = -0.35
    A[2-1, 7-1] = -0.35
    A[2-1, 8-1] = -0.33
    A[3-1, 1-1] = -0.40
    A[3-1, 2-1] = -0.26
    A[3-1, 3-1] = -0.33
    A[3-1, 4-1] = -0.40
    A[3-1, 5-1] = -0.32
    A[3-1, 6-1] = -0.22
    A[3-1, 7-1] = -0.33
    A[3-1, 8-1] = -0.42
    A[4-1, 1-1] = -0.50
    A[4-1, 2-1] = -0.37
    A[4-1, 3-1] = -0.40
    A[4-1, 4-1] = -0.53
    A[4-1, 5-1] = -0.48
    A[4-1, 6-1] = -0.32
    A[4-1, 7-1] = -0.42
    A[4-1, 8-1] = -0.39
    A[5-1, 1-1] = -0.49
    A[5-1, 2-1] = -0.38
    A[5-1, 3-1] = -0.32
    A[5-1, 4-1] = -0.48
    A[5-1, 5-1] = -0.48
    A[5-1, 6-1] = -0.35
    A[5-1, 7-1] = -0.43
    A[5-1, 8-1] = -0.40
    A[6-1, 1-1] = -0.40
    A[6-1, 2-1] = -0.35
    A[6-1, 3-1] = -0.22
    A[6-1, 4-1] = -0.32
    A[6-1, 5-1] = -0.35
    A[6-1, 6-1] = -0.41
    A[6-1, 7-1] = -0.33
    A[6-1, 8-1] = -0.33
    A[7-1, 1-1] = -0.44
    A[7-1, 2-1] = -0.35
    A[7-1, 3-1] = -0.33
    A[7-1, 4-1] = -0.42
    A[7-1, 5-1] = -0.43
    A[7-1, 6-1] = -0.33
    A[7-1, 7-1] = -0.43
    A[7-1, 8-1] = -0.42
    A[8-1, 8-1] = -0.49
    A[8-1, 1-1] = -0.49
    A[8-1, 2-1] = -0.33
    A[8-1, 3-1] = -0.31
    A[8-1, 4-1] = -0.39
    A[8-1, 5-1] = -0.40
    A[8-1, 6-1] = -0.33
    A[8-1, 7-1] = -0.42
    A[8-1, 8-1] = -0.49

    plt.close('all')
    plt.imshow(A, vmin=-0.5, vmax=-0.2, cmap='binary_r', origin='lower')
    plt.xticks(np.arange(8), ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.yticks(np.arange(8), ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.savefig('/Users/evan/Code/Insight/plots/silhouette_distmap.pdf')
    plt.close('all')

    return



def plot_tsne(X, y, scenarios):
    #users = df.user_num.values
    plt.close('all')
    X_embedded = TSNE(learning_rate=20, perplexity=4, n_iter=100000, n_iter_without_progress=1000, init='random').fit(X)
    print(X_embedded.kl_divergence_)
    X_embedded = TSNE(learning_rate=20, perplexity=4, n_iter=100000, n_iter_without_progress=1000, init='random').fit_transform(X)
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

    plt.scatter(xs, ys, c=y, s=y+2, alpha=1)
    for i in range(len(xs)):
        plt.text(xs[i], ys[i], str(y[i]) + ' ' + str(scenarios[i]))

    plt.savefig('/Users/evan/Code/Insight/plots/tsne/tsne_test.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')

    return


def plot_roc(truneg, trupos):

    plt.close('all')
    plt.plot(1.0 - np.array(trueneg), truepos, color='b')
    plt.xlabel('Fraction Mis-identified as Imposters', fontsize=14)
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