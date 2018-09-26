import numpy as np
import pdb
import keystroke
import os
import pandas as pd
DATADIR = keystroke.DATADIR

chars = {}

ngraphs = [1, 2]
npop = [20, 10]


def keynum_to_ascii(keynums):
    ascii_str = [chr(item) for item in keynums]
    ascii_str = np.array(ascii_str)
    ascii_str[ascii_str == '\r'] = ' '
    #pdb.set_trace()
    return ascii_str


def key_freq():
    filename = os.path.join(DATADIR, 'raw_cleaned.csv')
    df = pd.read_csv(filename)
    freq = df.groupby(['key_num']).size()
   # print(np.sort(df.key_num.unique()))


def get_topgraphs():
    topgraphs = {}
    for i, n in enumerate(ngraphs):
        col = "graph{}".format(n)
        topgraphs[n] = df_keystroke.groupby([col]).size().sort_values()[::-1].index.tolist()
    return topgraphs


def get_scenarios_per_user(df_keystroke):
    df = df_keystroke[["user_num", "scenario_num"]].drop_duplicates().groupby(["user_num"]).size().to_frame('N_scenarios').reset_index()

    return df


def userfilter_Nscenarios(df_keystroke, nmin=3):
    df = get_scenarios_per_user(df_keystroke)
    users_good = df[df.N_scenarios > nmin].user_num.get_values()
    df = df_keystroke[df_keystroke.user_num.isin(users_good)]

    return df


def get_Nsamples(df_keystroke):
    grouped = df_keystroke.groupby(["user_num", "scenario_num"])
    Nsamples = grouped.ngroups

    return Nsamples


def bin_samples(df, nbins):
    df['bin_num'] = np.nan

    groups = df.groupby(['user_num', 'scenario_num']).groups

    di = {}
    for n in range(nbins):
        di[n] = np.array([])
    for j, inds in enumerate(groups.values()):
        length = int(len(inds)/nbins)
        for n in range(nbins):
            temp = np.array([inds[n*length:(n+1)*length]])
            if n == nbins - 1:
                temp = np.append(temp, inds[(n+1)*length + np.arange(len(inds) % nbins)])
            di[n] = np.append(di[n], temp)

    for n in range(nbins):
        df.loc[di[n], 'bin_num'] = n+1

    return df


def keynums_to_keylocs(key_nums):

    toprow = "QWERTYUIOP"
    midrow = "ASDFGHJKL'"
    botrow = "ZXCVBNM."
    space = " "
    toprow_num = np.array([ord(i) for i in toprow])
    midrow_num = np.array([ord(i) for i in midrow])
    botrow_num = np.array([ord(i) for i in botrow])

    leftcol = "QWERTASDFGZXCV"
    rightcol = "YUIOPHJKL'NM."
    leftcol_num = np.array([ord(i) for i in leftcol])
    rightcol_num = np.array([ord(i) for i in rightcol])

    ignore = np.append(np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]), np.arange(90, 256, 1))

    allkeys = np.arange(0, 256, 1)
    hloc = np.array([np.nan]*len(allkeys))
    vloc = np.array([np.nan]*len(allkeys))
    for num in allkeys:
        if num in toprow_num:
            vloc[num] = 2
        elif num in midrow_num:
            vloc[num] = 1
        elif num in botrow_num:
            vloc[num] = 0
        elif num in ignore:
            vloc[num] = -1
        else:
            vloc[num] = num

        if num in leftcol_num:
            hloc[num] = 0
        elif num in rightcol_num:
            hloc[num] = 1
        elif num in ignore:
            hloc[num] = -1
        else:
            hloc[num] = num

    hlocs = [int(hloc[i]) for i in key_nums]
    vlocs = [int(vloc[i]) for i in key_nums]

    return hlocs, vlocs


