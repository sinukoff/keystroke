from nltk import ngrams
from keystroke import data_prep
import pandas as pd
import numpy as np
import pdb
import keystroke
import os

DATADIR = keystroke.DATADIR

ngraphs = [1, 2]
npop = [20, 10]

def get_features():
    df_keystroke = data_prep.read_cleaned()
    for n in ngraphs:
        df_keystroke = get_ngraph(df_keystroke, n)
    df_features = keystroke_to_user_df(df_keystroke)

    filename = os.path.join(DATADIR, 'features.csv')
    df_features.to_csv(filename)


    return


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
    for i in np.arange(0, n):
        ngraph[col1] = ngraph[col1] + '_' + df.key_num.shift(i)
        ngraph[col3] = ngraph[col3] + df.tdwell.shift(i)

    ngraph[col2] -= ngraph[col3]
    ngraph[col1] = ngraph[col1].str[1:]
    df = df.join(ngraph)

    return df

# for each user, for top 20 most common characters, get mean, MAD, and skew of duration


def keystroke_to_user_df(df_keystroke):

    topgraphs = {}
    for i, n in enumerate(ngraphs):
        col = "graph{}".format(n)
        topgraphs[n] = df_keystroke.groupby([col]).size().sort_values()[::-1].iloc[0:npop[i]].index.tolist()
    inds = df_keystroke[['user_num', 'scenario_num']].drop_duplicates().set_index(['user_num', 'scenario_num'])
    df = pd.DataFrame(index=inds.index)
    samples = df_keystroke.groupby(['user_num', 'scenario_num'])

    for n in topgraphs.keys():
        print(n)
        newcols = []
        suffixes = []
        for j in range(len(topgraphs[n])):
            suffix = '{}'.format(topgraphs[n][j])
            suffixes.append(suffix)
            suffixes.append(suffix)
            suffixes.append(suffix)
            suffixes.append(suffix)
            suffixes.append(suffix)
            newcols.append('count_{}'.format(suffix))
            newcols.append('med_flight_{}'.format(suffix))
            newcols.append('std_flight_{}'.format(suffix))
            newcols.append('med_dwell_{}'.format(suffix))
            newcols.append('std_dwell_{}'.format(suffix))
        for col in newcols:
            df[col] = np.nan

        for ind in df.index:
            print(ind)
            for i, col in enumerate(newcols):
                suffix = suffixes[i]
                df_temp = df_keystroke.loc[samples.groups[ind]]
                if col.startswith('med_flight'):
                    df.loc[ind, col] = df_temp[df_temp['graph'+str(n)].astype(str) == suffix]['tflight'+str(n)].median()
                elif col.startswith('std_flight'):
                    df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tflight' + str(n)].std()
                elif col.startswith('med_dwell'):
                    df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tdwell' + str(n)].median()
                elif col.startswith('std_dwell'):
                    df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tdwell' + str(n)].std()
                elif col.startswith('count'):
                    df.loc[ind, col] = len(df_temp[df_temp['graph' + str(n)].astype(str) == suffix])
               # grouped = df_keystroke[].groupby(['user_num', 'scenario_num']).filter(lambda x: str(x['tdwell'+str(n)]) == suffix)
                #.agg({'tdwell1': ['median', 'std']})

    for col in df.columns:
        if (col.startswith('med_flight') or col.startswith('std_flight')) and len(col.split('_')) == 3:
            df.drop(col, axis=1, inplace=True)
    return df


# def get_dwell_times(df_keystroke, npop):
#     keyfreq = df_keystroke.groupby(['key_num']).size()
#     topkeys = keyfreq.sort_values()[::-1].iloc[0:npop].index
#
#
#     users = df_keystroke.user_num.unique()
#     pdb.set_trace()
#     df_tdwell = pd.DataFrame(columns=['user_num', 'scenario_num', 'key_num'])
#    # for user in users:
#    #     for scene in scenarios:
#     #for i in range(npop):
#
#     df = df_keystroke.copy()
#     df.set_index(['user_num', 'scenario_num', 'key_num'], inplace=True)
#     pdb.set_trace()
#     df.sortlevel(inplace=True)
#
#     idx = pd.IndexSlice
#     #for i in df.index:
#    # pdb.set_trace()
#    # df.loc[idx[:, 7, [188, 190]], :]
#     df['tdwell_arr'] = np.nan
#     dwell = []
#     for ind in df.index:
#         print(ind)
#         pdb.set_trace()
#       #  df.tdwell_array =
#        # df.loc[idx[:, 7, [188, 190]], :]
#
#         # for scene in df.index:
#         #     for key in topkeys:
#         #         t_dwell.loc[user, "tdwell_{}".format(key)] = df_keystrokes
#
# #    for i in sorted
#
#     return

# repeat with top 20 di-graphs



