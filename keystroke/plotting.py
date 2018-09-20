import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keystroke import data_prep
from statsmodels.robust.scale import mad as MAD

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

