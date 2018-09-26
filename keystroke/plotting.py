import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keystroke import data_prep, features
from statsmodels.robust.scale import mad as MAD
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

