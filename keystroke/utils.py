import numpy as np
import keystroke
import os
import pandas as pd

DATADIR = keystroke.DATADIR
ngraphs = keystroke.ngraphs


def keynum_to_ascii(keynums):
    ascii_str = [chr(item) for item in keynums]
    ascii_str = np.array(ascii_str)
    ascii_str[ascii_str == '\r'] = ' '
    return ascii_str


def key_freq():
    filename = os.path.join(DATADIR, 'raw_cleaned.csv')
    df = pd.read_csv(filename)
    freq = df.groupby(['key_num']).size()
   # print(np.sort(df.key_num.unique()))


def get_topgraphs(df_keystroke):
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

    key_nums = key_nums.astype(int)

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

   # pdb.set_trace()
    hlocs = [int(hloc[i]) for i in key_nums.values]
    vlocs = [int(vloc[i]) for i in key_nums.values]
    zones = np.array([np.nan]*len(vlocs))


    df = pd.DataFrame({'hloc': hlocs, 'vloc': vlocs, 'zone': zones})

    zone_list = [[0, 0, 1],
                 [0, 1, 2],
                 [0, 2, 3],
                 [1, 0, 4],
                 [1, 1, 5],
                 [1, 2, 6],
                 [8, 8, 8],
                [32, 32, 32]
                ]


    for zone in zone_list:
        ind = df[(df.hloc == zone[0]) & (df.vloc == zone[1])].index
        df.loc[ind, 'zone'] = zone[2]
    df.zone.fillna(-1, inplace=True)
    df.zone = df.zone.astype(int)

    return df.hloc.values, df.vloc.values, df.zone.values


def get_nloc(df, n):
    hcol = "hloc{}".format(n)
    vcol = "vloc{}".format(n)
    zonecol = "zone{}".format(n)

    nloc = pd.DataFrame(index=df.index, columns=[hcol, vcol, zonecol])

    nloc[hcol] = df.hloc
    nloc[vcol] = df.vloc
    nloc[zonecol] = df.zone

    nloc[vcol] = df['vloc'].shift(n-1)
    nloc[hcol] = df['hloc'].shift(n-1)
    nloc[zonecol] = df['zone'].shift(n-1)

    #for i in np.arange(1, n)[::-1]:
        #nloc[vcol] = pd.Series(np.array(pd.concat([nloc[vcol], df['vloc'].shift(1)], axis=1).get_values()).tolist())
        #nloc[hcol] = pd.Series(np.array(pd.concat([nloc[hcol], df['hloc'].shift(1)], axis=1).get_values()).tolist())

    df = df.join(nloc)

    return df



def get_ngraph(df, n):
    col1 = "graph{}".format(n)
    col2 = "tflight{}".format(n)
    col3 = "tdwell{}".format(n)
    df.key_num = df.key_num.astype(str)
    ngraph = pd.DataFrame(index=df.index, columns=[col1, col2])
    ngraph[col1] = ''
    ngraph[col2] = df.tstop - df.tstart.shift(n-1)
    ngraph[col3] = 0

    #todo-what to do about ngrams that span multiple scenarios/users. These are likely to be trimmed out based on time between keys, but maybe not. maybe trim any trigrams > 300
    for i in np.arange(0, n)[::-1]:
        ngraph[col1] = ngraph[col1] + '_' + df.key_num.shift(i)
        ngraph[col3] = ngraph[col3] + df.tdwell.shift(i)

    ngraph[col2] -= ngraph[col3]
    ngraph[col1] = ngraph[col1].str[1:]
    df = df.join(ngraph)

    df.key_num = df.key_num.astype(int)

    return df


def get_missed_frac():

    response_filename = os.path.join(DATADIR, 'responses.csv')

    df_text = pd.read_csv(
        response_filename,
        encoding="ISO-8859-1"
    )

    df_missed = pd.DataFrame(columns=['user_num', 'scenario_num', 'nobs', 'ndel', 'ntrue', 'missedfrac'])
    user = []
    scenario = []
    nobs = []
    ndel = []
    ntrue = []

    for n, i in enumerate(df[['userId', 'scenarioId']].drop_duplicates().values):
        print(n)

        try:
            ntrue.append(len(df_text[(df_text.userId == i[0]) & (df_text.scenarioId == i[1])].answers.iloc[0]))
        except IndexError:
            print('missed', n)
            continue

        user.append(df.loc[df.userId == i[0], 'user_num'].iloc[0])
        scenario.append(df.loc[df.scenarioId == i[1], 'scenario_num'].iloc[0])
        nobs.append(len(df[(df.userId == i[0]) & (df.scenarioId == i[1])]))
        ndel.append(len(df[(df.userId == i[0]) & (df.scenarioId == i[1]) & (df.key_num == 8)]))

    df_missed.user_num = user
    df_missed.scenario_num = scenario
    df_missed.nobs = nobs
    df_missed.ndel = ndel
    df_missed.ntrue = ntrue
    df_missed.missedfrac = (df_missed.ntrue + df_missed.ndel - df_missed.nobs)/(df_missed.ntrue + df_missed.ndel)
    df_missed.to_csv(os.path.join(DATADIR, 'missedfrac.csv'))
