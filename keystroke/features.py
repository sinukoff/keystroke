from keystroke import data_prep, utils, plotting
import pandas as pd
import numpy as np
import keystroke
import os
from statsmodels.robust.scale import mad as MAD
import itertools
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score


DATADIR = keystroke.DATADIR
zones = keystroke.zones


def main():
    df_keystroke = data_prep.read_cleaned()
    df = keystrokes_to_features(df_keystroke)
    df.to_csv(os.path.join(DATADIR, 'features.csv'), index=False)


def preprocess(df, plot=False):

    X = impute(df[df.columns[3:]], allow_null=False)
    X = StandardScaler().fit_transform(X)
    df_out = pd.DataFrame(X, index=[df.user_num, df.scenario_num, df.bin_num]).reset_index()

    if plot:
        plotting.plot_tsne(X, y=df_out.user_num.values, scenarios=df_out.scenario_num.values)

    return df_out


def read_features():
    filename = os.path.join(DATADIR, 'features.csv')
    df = pd.read_csv(filename)

    return df



def keystrokes_to_features(df_keystroke):

    users = df_keystroke.drop_duplicates(['user_num', 'scenario_num', 'bin_num'])[['user_num', 'scenario_num', 'bin_num']]
    users = users.groupby('user_num').size().to_frame('cnt')
    df_missed = pd.read_csv(os.path.join(DATADIR, 'missedfrac.csv'))
    #ind = df_missed[df_missed.missedfrac.between(-0.01, 0.018) & (~df_missed.user_num.isin([77, 271,  387, 395]))].index

   # np.random.RandomState(40)
   # ind = df_missed[df_missed.missedfrac.between(0.6, 0.79)]
   # ind = ind.drop_duplicates(['user_num', 'scenario_num']).iloc[np.random.randint(0, len(ind), size=22)].index
   # ind = ind[~ind.user_num.isin([])].index

   # ind = df_missed[df_missed.user_num.isin([5, 33, 40, 41, 42, 47, 54, 57, 61, 65, 68, 77, 87])].index
   # pdb.set_trace()

    ind = df_missed[(df_missed.missedfrac.between(0.00, 0.10)) & ((df_missed.user_num.isin([33, 41, 160, 212, 377, 395, 426, 460])) | (df_missed.user_num.isin([42, 65, 97, 101, 123, 154, 206, 309, 326, 400, 492, 494, 517])) )].index
    df_missed = df_missed.loc[ind]
  #  print(df_missed)

    keep = df_missed[['user_num', 'scenario_num']].values
    indkeep = np.array([])
    for i in keep:
        print(i, len(keep))
        if i[0] not in [241]:
            indkeep = np.append(indkeep, df_keystroke[(df_keystroke.user_num == i[0]) & (df_keystroke.scenario_num == i[1])].index)

    #df_keystroke = df_keystroke[df_keystroke.user_num.isin(users)]
    df_keystroke = df_keystroke.loc[indkeep]
    inds = df_keystroke[['user_num', 'scenario_num', 'bin_num']].drop_duplicates().set_index(['user_num', 'scenario_num', 'bin_num']).index
    df = pd.DataFrame(index=inds)
    combos = list(itertools.product(zones, repeat=2))
    zonefreq = df_keystroke.groupby(['zone', 'zone2']).size().sort_values(ascending=False).to_frame('zonefreq').reset_index().iloc[0:20]

    zonekeep = []
    combos_new = pd.Series(combos)
    for i in range(len(combos_new)):
        if len(zonefreq[(zonefreq.zone == combos_new[i][0]) & (zonefreq.zone2 == combos_new[i][1])]) > 0:
            zonekeep.append(i)

    #pdb.set_trace()
    combos = combos_new[zonekeep].tolist()

   # pdb.set_trace()
    for zone1, zone2 in combos:
        newcol = 'zoneflight_med_{}_{}'.format(zone1, zone2)
        df[newcol] = np.nan
      #  newcol = 'zoneflight_std_{}_{}'.format(zone1, zone2)
      #  df[newcol] = np.nan
        newcol = 'zonedwell_med_{}_{}'.format(zone1, zone2)
        df[newcol] = np.nan
       # newcol = 'zonedwell_std_{}_{}'.format(zone1, zone2)
      #  df[newcol] = np.nan
        newcol = 'zonedwell2_med_{}_{}'.format(zone1, zone2)
        df[newcol] = np.nan
       # newcol = 'zonedwell2_std_{}_{}'.format(zone1, zone2)
       # df[newcol] = np.nan

       # newcol = 'zoneflight_min_{}_{}'.format(zone1, zone2)
       # df[newcol] = np.nan


       # for zone1, zone2 in combos:
       #     ind = df_keystroke[(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])].index


    for i, samp in enumerate(df.index):
        print("{}/{}".format(i+1, len(df.index)))
        for zone1, zone2 in combos:
            ind = df_keystroke[(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])].index

            col = 'zoneflight_med_{}_{}'.format(zone1, zone2)
            df.loc[samp, col] = np.median(df_keystroke.loc[ind, 'tflight2'])

           # col = 'zoneflight_std_{}_{}'.format(zone1, zone2)
           # df.loc[samp, col] = MAD(df_keystroke.loc[ind, 'tflight2'])

            col = 'zonedwell_med_{}_{}'.format(zone1, zone2)
            df.loc[samp, col] = np.median(df_keystroke.loc[ind, 'tdwell'])

           # col = 'zonedwell_std_{}_{}'.format(zone1, zone2)
           # df.loc[samp, col] = MAD(df_keystroke.loc[ind, 'tdwell'])

            col = 'zonedwell2_med_{}_{}'.format(zone1, zone2)
            df.loc[samp, col] = np.median(df_keystroke.loc[ind, 'tdwell2'])

            #col = 'zonedwell2_std_{}_{}'.format(zone1, zone2)
            #df.loc[samp, col] = MAD(df_keystroke.loc[ind, 'tdwell2'])

           # col = 'zoneflight_min_{}_{}'.format(zone1, zone2)
           # df.loc[samp, col] = np.min(df_keystroke.loc[ind, 'tflight2'])



            # df.loc[samp, col] = np.median(df_keystroke['tflight2'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )
            # col = 'zoneflight_std_{}_{}'.format(zone1, zone2)
            # df.loc[samp, col] = np.std(df_keystroke['tflight2'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )
            # col = 'zonedwell_med_{}_{}'.format(zone1, zone2)
            # df.loc[samp, col] = np.median(df_keystroke['tdwell'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )
            # col = 'zonedwell_std_{}_{}'.format(zone1, zone2)
            # df.loc[samp, col] = np.std(df_keystroke['tdwell'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )
            # col = 'zonedwell2_med_{}_{}'.format(zone1, zone2)
            # df.loc[samp, col] = np.median(df_keystroke['tdwell2'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )
            # col = 'zonedwell2_std_{}_{}'.format(zone1, zone2)
            # df.loc[samp, col] = np.std(df_keystroke['tdwell2'][(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == samp[0]) & (df_keystroke.scenario_num == samp[1]) & (df_keystroke.bin_num == samp[2])] )


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



def plot_tsne(df):
    users = df.user_num.values
    scenarios = df.scenario_num.values

    plotting.tsne(df, users=users, scenarios=scenarios)



if __name__ == "__main__":
    main()




