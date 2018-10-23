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
import pdb


DATADIR = keystroke.DATADIR

def main():
    df_keystroke = data_prep.read_cleaned()
    df_keystroke = keystroke_missedfrac_trim(df_keystroke)
    df = keystrokes_to_features(df_keystroke)
    df.to_csv(os.path.join(DATADIR, 'features.csv'), index=False)


def preprocess(df, plot=False):

    X = impute(df[df.columns[3:]], allow_null=False)
    X = StandardScaler().fit_transform(X)
    df_out = pd.DataFrame(X, index=[df.user_num, df.scenario_num, df.bin_num]).reset_index()

    return df_out


def read_features():
    filename = os.path.join(DATADIR, 'features.csv')
    df = pd.read_csv(filename)

    return df


def keystroke_missedfrac_trim(df_keystroke):
    df_missed = pd.read_csv(os.path.join(DATADIR, 'missedfrac.csv'))

    # users = df_keystroke.drop_duplicates(['user_num', 'scenario_num', 'bin_num'])[['user_num', 'scenario_num', 'bin_num']]
    # users = users.groupby('user_num').size().to_frame('cnt')
   # ind = df_missed[df_missed.missedfrac.between(-0.01, 0.018) & (~df_missed.user_num.isin([77, 271,  387, 395]))].index


    #ind = df_missed[df_missed.missedfrac.between(0.0, 0.02) & (~df_missed.user_num.isin([77, 271, 387, 395]))].index
   # ind = df_missed[df_missed.missedfrac.between(0.0, 0.02)]
    #pdb.set_trace()
    #ind = ind.drop_duplicates(['user_num']).index
   # pdb.set_trace()
   # np.random.RandomState(40)
    # ind = df_missed[df_missed.missedfrac.between(0.6, 0.79)]
   # pdb.set_trace()
    #ind = ind.drop_duplicates(['user_num', 'scenario_num']).iloc[np.random.randint(0, len(ind), size=23)].index
    # ind = ind[~ind.user_num.isin([])].index

    #ind = df_missed[df_missed.user_num.isin([5, 33, 40, 41, 42, 47, 54, 57, 61, 65, 68, 77, 87])].index
    # pdb.set_trace()

    # pdb.set_trace()
    df_accurate = df_missed.query('missedfrac.between(0.00, 0.10)')
    df_multiscen_accurate = df_accurate.groupby(['user_num']).filter(lambda group: len(group) > 1)
   # df_multiscen_accurate.query('user_num != 241', inplace=True)

    # ind = df_missed[(df_missed.missedfrac.between(0.00, 0.10)) & ((df_missed.user_num.isin([33, 41, 160, 212, 377, 395, 426, 460])) | (df_missed.user_num.isin([42, 65, 97, 101, 123, 154, 206, 309, 326, 400, 492, 494, 517])) )].index

    #  print(df_missed)

    keep = df_multiscen_accurate[['user_num', 'scenario_num']].values
   # keep = df_missed.loc[ind, ['user_num', 'scenario_num']].values

    indkeep = np.array([])
    for i in keep:
        print(i, len(keep))
        indkeep = np.append(indkeep,
                            df_keystroke[(df_keystroke.user_num == i[0]) & (df_keystroke.scenario_num == i[1])].index)
    # df_keystroke = df_keystroke[df_keystroke.user_num.isin(users)]
    df_keystroke = df_keystroke.loc[indkeep]

    return df_keystroke


def keystrokes_to_features(df_keystroke, zones, nzonesmax, featurecols, med, mad):

    #Set up features dataframe
    #pdb.set_trace()
    inds = df_keystroke[['user_num', 'scenario_num', 'bin_num']].drop_duplicates().set_index(['user_num', 'scenario_num', 'bin_num']).index
    df = pd.DataFrame(index=inds)

    #Get overall frequencies of transitions between zones
    combos = list(itertools.product(zones, repeat=2))
    zonefreq = df_keystroke.groupby(['zone', 'zone2']).size().sort_values(ascending=False).to_frame('zonefreq').reset_index().iloc[0:nzonesmax]

    #Keep the nzonesmax most common zone transitions
    zonekeep = []
    combos_new = pd.Series(combos)

    for i in range(len(combos_new)):
        if len(zonefreq[(zonefreq.zone == combos_new[i][0]) & (zonefreq.zone2 == combos_new[i][1])]) > 0:
            zonekeep.append(i)

    combos = combos_new[zonekeep].tolist()

    # Create feature columns
    for zone1, zone2 in combos:
        for col in featurecols:
            if med:
                newcol = '{}_med_zone_{}_{}'.format(col, zone1, zone2)
                df[newcol] = np.nan
            if mad:
                newcol = '{}_mad_zone_{}_{}'.format(col, zone1, zone2)
                df[newcol] = np.nan

    # Compute values for feature columns
    for i, ind in enumerate(df.index):
        print("{}/{}".format(i+1, len(df.index)))
        for zone1, zone2 in combos:
            ind_key = df_keystroke[(df_keystroke.zone == zone1) & (df_keystroke.zone2 == zone2) & (df_keystroke.user_num == ind[0]) & (df_keystroke.scenario_num == ind[1]) & (df_keystroke.bin_num == ind[2])].index

            for col in featurecols:
                if med:
                    colname = '{}_med_zone_{}_{}'.format(col, zone1, zone2)
                    df.loc[ind, colname] = np.nanmedian(df_keystroke.loc[ind_key, col])
                if mad:
                    colname = '{}_mad_zone_{}_{}'.format(col, zone1, zone2)
                    df.loc[ind, colname] = MAD(df_keystroke.loc[ind_key, col])

    df.reset_index(inplace=True)

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



if __name__ == "__main__":
    main()




