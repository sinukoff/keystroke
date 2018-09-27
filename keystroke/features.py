from nltk import ngrams
from keystroke import data_prep, utils
import pandas as pd
import numpy as np
import pdb
import keystroke
import os
from statsmodels.robust.scale import mad as MAD
import itertools
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATADIR = keystroke.DATADIR

ngraphs = [1, 2]
npop = [20, 10]
zones = [1, 2, 3, 4, 5, 6, 8, 32]
bins_per_scen = 4

def get_features():
    df_keystroke = data_prep.read_cleaned()
    df = keystrokes_to_features(df_keystroke)

    # IMPUTE WITHIN EACH USER
    for user in df.user_num.unique():
        ind = df[df.user_num == user].index
        df.loc[ind, df.columns[3:]] = impute(df.loc[ind, df.columns[3:]], allow_null=True)

    # IMPUTE WITHIN ALL USERS
    df[df.columns[3:]] = impute(df[df.columns[3:]], allow_null=False)

    df[df.columns[3:]] = StandardScaler().fit_transform(df[df.columns[3:]])

    #df = df[['user_num', 'scenario_num', 'bin_num', 'zoneflight_med_1_3', 'zoneflight_med_3_3']]
    df195 = df[df.user_num==195]
    df299 = df[df.user_num == 299]
    plt.close('all')
    plt.plot(df195['zoneflight_med_1_3'], df195['zoneflight_med_3_3'], 'o', color='r')
    plt.plot(df299['zoneflight_med_1_3'], df299['zoneflight_med_3_3'], 'o', color='b')
    plt.savefig('/Users/evan/Code/Insight/plots/test.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')
    users = df.user_num.values

    pca = PCA(n_components=10)
    #df = pca.fit_transform(df[df.columns[3:]])
    #pdb.set_trace()
    #tsne(df, users=users)
    tsne(df[df.columns[3:]], users=users)
    pdb.set_trace()

    df_features1 = keystroke_to_user_df(df_keystroke)

  #  df_features1 = keystroke_to_user_df(df_keystroke)
   #df_features2 = keystroke_to_feature_df(df_keystroke)

   # filename2 = os.path.join(DATADIR, 'features2.csv')
   # df_features1.to_csv(filename1)
   # filename = os.path.join(DATADIR, 'features2.csv')
   # df_features2.to_csv(filename)


    return


def read_features():
    filename = os.path.join(DATADIR, 'features.csv')
    df = pd.read_csv(filename)

    return df



# def keystroke_to_user_df(df_keystroke):
#
#     topgraphs = {}
#     for i, n in enumerate(ngraphs):
#         col = "graph{}".format(n)
#         topgraphs[n] = df_keystroke.groupby([col]).size().sort_values()[::-1].iloc[0:npop[i]].index.tolist()
#     inds = df_keystroke[['user_num', 'scenario_num', 'bin_num']].drop_duplicates().set_index(['user_num', 'scenario_num', 'bin_num'])
#     df = pd.DataFrame(index=inds.index)
#     samples = df_keystroke.groupby(['user_num', 'scenario_num', 'bin_num'])
#
#     for n in topgraphs.keys():
#         print(n)
#         newcols = []
#         suffixes = []
#         for j in range(len(topgraphs[n])):
#             suffix = '{}'.format(topgraphs[n][j])
#             suffixes.append(suffix)
#             suffixes.append(suffix)
#             suffixes.append(suffix)
#             suffixes.append(suffix)
#             suffixes.append(suffix)
#             newcols.append('count_{}'.format(suffix))
#             newcols.append('med_flight_{}'.format(suffix))
#             newcols.append('std_flight_{}'.format(suffix))
#             newcols.append('med_dwell_{}'.format(suffix))
#             newcols.append('std_dwell_{}'.format(suffix))
#         for col in newcols:
#             df[col] = np.nan
#
#         for ind in df.index:
#             print(ind)
#             for i, col in enumerate(newcols):
#                 suffix = suffixes[i]
#                 df_temp = df_keystroke.loc[samples.groups[ind]]
#                 if col.startswith('med_flight'):
#                     df.loc[ind, col] = df_temp[df_temp['graph'+str(n)].astype(str) == suffix]['tflight'+str(n)].median()
#                 elif col.startswith('std_flight'):
#                     df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tflight' + str(n)].std()
#                 elif col.startswith('med_dwell'):
#                     df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tdwell' + str(n)].median()
#                 elif col.startswith('std_dwell'):
#                     df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tdwell' + str(n)].std()
#                 elif col.startswith('count'):
#                     df.loc[ind, col] = len(df_temp[df_temp['graph' + str(n)].astype(str) == suffix])
#                # grouped = df_keystroke[].groupby(['user_num', 'scenario_num']).filter(lambda x: str(x['tdwell'+str(n)]) == suffix)
#                 #.agg({'tdwell1': ['median', 'std']})
#
#     for col in df.columns:
#         if (col.startswith('med_flight') or col.startswith('std_flight')) and len(col.split('_')) == 3:
#             df.drop(col, axis=1, inplace=True)
#     return df



# def keystroke_to_feature_df(df_keystroke):
#
#     topgraphs = {}
#     for i, n in enumerate(ngraphs):
#         col = "graph{}".format(n)
#         topgraphs[n] = df_keystroke.groupby([col]).size().sort_values()[::-1].index.tolist()
#
#
#    # inds = df_keystroke[['user_num', 'scenario_num', 'graph1']].drop_duplicates().set_index(['user_num', 'scenario_num', 'graph1'])
#
#
#     in_all = {}
#
#     #1-graphs
#    # df1 = pd.pivot_table(df_keystroke, index=["user_num", "scenario_num", "graph1"],
#    #                      values=['tdwell1'],
#    #                      aggfunc=[len]).add_prefix('count_')
#
#
#     df_keystroke = utils.userfilter_Nscenarios(df_keystroke, nmin=3)
#     # df = df_keystroke[["user_num", "scenario_num"]].drop_duplicates().groupby(["user_num"]).size().to_frame('N_scenarios').reset_index()
#     # users_good = df[df.N_scenarios > 3].user_num.get_values()
#     # df_keystroke = df_keystroke[df_keystroke.user_num.isin(users_good)]
#
#     # grouped = df_keystroke.groupby(["user_num", "scenario_num"])
#     # Ngroups = grouped.ngroups
#
#     #pdb.set_trace()
#     #users_good = N_scenarios[]
#
#     # 1-GRAPHS
#     #get keys used by all users in all scenarios
#     df0 = df_keystroke.groupby(["user_num", "scenario_num", "graph1"]).size().to_frame('cnt').reset_index()
#     df00 = df0.groupby(["graph1"]).size().to_frame('cnt')
#     df000 = df00[df00['cnt'] == Ngroups]
#     keys_in_all = np.array(df000.index)
#
#     # Of these, how many are used more than N times by all users
#     df00 = df0.groupby(["graph1"])['cnt'].min()
#     keys_in_all_2 = np.array(df00[df00 > 1].index)
#     in_all[1] = np.intersect1d(keys_in_all, keys_in_all_2)
#
#     # 2-GRAPHS
#     df0 = df_keystroke.groupby(["user_num", "scenario_num", "graph2"]).size().to_frame('cnt').reset_index()
#     df00 = df0.groupby(["graph2"]).size().to_frame('cnt')
#     df000 = df00[df00['cnt'] == Ngroups]
#     keys_in_all = np.array(df000.index)
#
#     df00 = df0.groupby(["graph2"])['cnt'].min()
#     keys_in_all_2 = np.array(df00[df00 > 1].index)
#     in_all[2] = np.intersect1d(keys_in_all, keys_in_all_2)
#
#     pdb.set_trace()
#
#     df = df_keystroke[(df_keystroke.graph1.isin(in_all[1])) | (df_keystroke.graph2.isin(in_all[2]))]
#
#     pdb.set_trace()
#
#
#
#     groups.size().to_frame('size').reset_index()
#     pdb.set_trace()
#     df2 = pd.pivot_table(df_keystroke, index=["user_num", "scenario_num", "graph1"], values=['tdwell1'],
#                             aggfunc=[np.median]).add_prefix('median_')
#
#     df = pd.merge(df1, df2, on=["user_num", "scenario_num", "graph1"])
#
#     df1 = pd.pivot_table(df_keystroke, index=["user_num", "scenario_num", "graph1"], values=['tdwell1'],
#                             aggfunc=[MAD]).add_prefix('mad_')
#
#     df_1 = pd.merge(df, df1, on=["user_num", "scenario_num", "graph1"]).reset_index()
#
#
#    #2-graphs
#
#     df1 = pd.pivot_table(df_keystroke, index=["user_num", "scenario_num", "graph2"], values=['tdwell2'],
#                             aggfunc=[len]).add_prefix('count_')
#
#     df2 = pd.pivot_table(df_keystroke, index=["user_num", "scenario_num", "graph2"], values=['tdwell2', 'tflight2'],
#                             aggfunc=[np.median]).add_prefix('median_')
#
#     df = pd.merge(df1, df2, on=["user_num", "scenario_num", "graph2"])
#
#     df1 = pd.pivot_table(df_keystroke, index=["user_num", "scenario_num", "graph2"], values=['tdwell2', 'tflight2'],
#                             aggfunc=[MAD]).add_prefix('mad_')
#
#     df_2 = pd.merge(df, df1, on=["user_num", "scenario_num", "graph2"]).reset_index()
#
#
#     pdb.set_trace()
#
#     samples = df_keystroke.groupby(['user_num', 'scenario_num'])
#     df_1graphs = samples.size()[::-1].unstack()
#
#
#     pdb.set_trace()
#     for n in topgraphs.keys():
#         print(n)
#         newcols = []
#         suffixes = []
#         for j in range(len(topgraphs[n])):
#             suffix = '{}'.format(topgraphs[n][j])
#             suffixes.append(suffix)
#             suffixes.append(suffix)
#             suffixes.append(suffix)
#             suffixes.append(suffix)
#             suffixes.append(suffix)
#             newcols.append('count_{}'.format(suffix))
#             newcols.append('med_flight_{}'.format(suffix))
#             newcols.append('std_flight_{}'.format(suffix))
#             newcols.append('med_dwell_{}'.format(suffix))
#             newcols.append('std_dwell_{}'.format(suffix))
#         for col in newcols:
#             df[col] = np.nan
#
#         for ind in df.index:
#             print(ind)
#             for i, col in enumerate(newcols):
#                 suffix = suffixes[i]
#                 df_temp = df_keystroke.loc[samples.groups[ind]]
#                 if col.startswith('med_flight'):
#                     df.loc[ind, col] = df_temp[df_temp['graph'+str(n)].astype(str) == suffix]['tflight'+str(n)].median()
#                 elif col.startswith('std_flight'):
#                     df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tflight' + str(n)].std()
#                 elif col.startswith('med_dwell'):
#                     df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tdwell' + str(n)].median()
#                 elif col.startswith('std_dwell'):
#                     df.loc[ind, col] = df_temp[df_temp['graph' + str(n)].astype(str) == suffix]['tdwell' + str(n)].std()
#                 elif col.startswith('count'):
#                     df.loc[ind, col] = len(df_temp[df_temp['graph' + str(n)].astype(str) == suffix])
#                # grouped = df_keystroke[].groupby(['user_num', 'scenario_num']).filter(lambda x: str(x['tdwell'+str(n)]) == suffix)
#                 #.agg({'tdwell1': ['median', 'std']})
#
#     for col in df.columns:
#         if (col.startswith('med_flight') or col.startswith('std_flight')) and len(col.split('_')) == 3:
#             df.drop(col, axis=1, inplace=True)
#     return df
#

def keystrokes_to_features(df_keystroke):

    users = df_keystroke.drop_duplicates(['user_num', 'scenario_num', 'bin_num'])[['user_num', 'scenario_num', 'bin_num']]
    users = users.groupby('user_num').size().to_frame('cnt')
    users = users[users.cnt > 6*bins_per_scen].index.tolist()

    df_keystroke = df_keystroke[df_keystroke.user_num.isin(users)]
    inds = df_keystroke[['user_num', 'scenario_num', 'bin_num']].drop_duplicates().set_index(['user_num', 'scenario_num', 'bin_num']).index
    df = pd.DataFrame(index=inds)
    combos = list(itertools.combinations_with_replacement(zones, 2))

    for zone1, zone2 in combos:
        newcol = 'zoneflight_med_{}_{}'.format(zone1, zone2)
        df[newcol] = np.nan
        newcol = 'zoneflight_std_{}_{}'.format(zone1, zone2)
        df[newcol] = np.nan
        newcol = 'zonedwell_med_{}_{}'.format(zone1, zone2)
        df[newcol] = np.nan
        newcol = 'zonedwell_std_{}_{}'.format(zone1, zone2)
        df[newcol] = np.nan




    for samp in df.index:
        for zone1, zone2 in combos:
            col = 'zoneflight_med_{}_{}'.format(zone1, zone2)
            df.loc[samp, col] = np.median(df_keystroke['tflight2'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )
            col = 'zoneflight_std_{}_{}'.format(zone1, zone2)
            df.loc[samp, col] = np.std(df_keystroke['tflight2'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )
            col = 'zonedwell_med_{}_{}'.format(zone1, zone2)
            df.loc[samp, col] = np.median(df_keystroke['tdwell'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )
            col = 'zonedwell_std_{}_{}'.format(zone1, zone2)
            df.loc[samp, col] = np.std(df_keystroke['tdwell'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )


    cols = df.columns
    df = df.reset_index()

    combos = list(itertools.combinations(cols, 2))

    # for col1, col2 in combos:
    #     plt.close('all')
    #     plt.plot(df[col1][df.user_num == 195], df[col2][df.user_num == 195], 'o', color='r')
    #     plt.plot(df[col1][df.user_num == 299], df[col2][df.user_num == 299], 'o', color='b')
    #     plt.plot(df[col1][df.user_num == 252], df[col2][df.user_num == 252], 'o', color='g')
    #     plt.plot(df[col1][df.user_num == 440], df[col2][df.user_num == 440], 'o', color='k')
    #     plt.plot(df[col1][df.user_num == 517], df[col2][df.user_num == 517], 'o', color='gray')
    #     savefile = '/Users/evan/Code/Insight/plots/feature_test/{}_{}.pdf'.format(col1, col2)
    #     plt.savefig(savefile, bbox_inches='tight', pad_inches=0.01)
    #     plt.close('all')


    return df


def impute(df_features, allow_null=True):

    badcols = []
    for i, col in enumerate(df_features.columns):
        if len(df_features[df_features[col].isnull()]) == len(df_features):
            badcols.append(i)

    imp = SimpleImputer(strategy='median')
    df_imputed = imp.fit_transform(df_features)

    if allow_null:
        for badcol in badcols:
            df_imputed = np.insert(df_imputed, badcol, np.nan, axis=1)

    return df_imputed


def tsne(df, users):
    plt.close('all')
    X_embedded = TSNE(learning_rate=20, perplexity=4*bins_per_scen, n_iter=100000, n_iter_without_progress=1000, init='random').fit_transform(df)
    xs = X_embedded[:, 0]
    ys = X_embedded[:, 1]
    xmin = np.argmin(xs)
    xmax = np.argmax(xs)
    ymin = np.argmax(ys)
    ymax = np.argmax(ys)

    xs = np.delete(xs, [xmin, xmax, ymin, ymax])
    ys = np.delete(ys, [xmin, xmax, ymin, ymax])
    users = np.delete(users, [xmin, xmax, ymin, ymax])

    plt.scatter(xs, ys, c=users, s=5*np.sqrt(users))
    plt.savefig('/Users/evan/Code/Insight/plots/tsne/tsne_test.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')




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



