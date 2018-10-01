import os
import pandas as pd
import keystroke
from keystroke import utils
import numpy as np
import pdb

bins_per_scen = 4
ngraphs = [1, 2]

DATADIR = keystroke.DATADIR


def main():
    df = read_raw()
  #  df = df[(df.userId == 'JXZdrt47ijLcFaHoM') & (df.scenarioId == 'qBPkXtZHbHn2XJZ5z')]
  #  pdb.set_trace()
    df = clean_raw(df)
    save_raw(df)

    return


def read_raw():
    filename = os.path.join(DATADIR, 'keystroke_raw.csv')
    df = pd.read_csv(filename)
    df = df
    return df


def clean_raw(df):
    df.rename(columns={'key': 'key_num',
                       'pushedDownAt': 'tstart',
                       'pushedUpAt': 'tstop'
                       },
              inplace=True)

    for col in df.columns:
        if col.startswith('Unnamed'):
            df.drop(columns=col, inplace=True)

    df.key_num = df.key_num.astype(int)
    df.scenarioId = df.scenarioId.astype('object')
    df.tstart = df.tstart.astype(float)
    df.tstop = df.tstop.astype(float)

    df['user_num'] = df.groupby('userId').ngroup()
    df['scenario_num'] = df.groupby('scenarioId').ngroup()
    df['key_ascii'] = utils.keynum_to_ascii(df.key_num)
   # pdb.set_trace()
    df['tdwell'] = df.tstop - df.tstart
    df = df.sort_values(by=['user_num', 'tstart']).drop_duplicates(['user_num', 'tstart', 'tstop']).reset_index(drop=True)
    groups = df.groupby(['user_num', 'scenario_num'])
  #  pdb.set_trace()
    # df_text = pd.read_csv(os.path.join(DATADIR, 'responses.csv'), encoding="ISO-8859-1")
    # missedfrac = []
    # df_missed = pd.DataFrame(columns=['user_num', 'scenario_num', 'nobs', 'ndel', 'ntrue', 'missedfrac'])
    # user = []
    # scenario = []
    # nobs = []
    # ndel = []
    # ntrue = []
    #
    # for n, i in enumerate(df[['userId', 'scenarioId']].drop_duplicates().values):
    #     print(n)
    #
    #     try:
    #         ntrue.append(len(df_text[(df_text.userId == i[0]) & (df_text.scenarioId == i[1])].answers.iloc[0]))
    #     except IndexError:
    #         print('missed', n)
    #         continue
    #
    #     user.append(df.loc[df.userId == i[0], 'user_num'].iloc[0])
    #     scenario.append(df.loc[df.scenarioId == i[1], 'scenario_num'].iloc[0])
    #     nobs.append(len(df[(df.userId == i[0]) & (df.scenarioId == i[1])]))
    #     ndel.append(len(df[(df.userId == i[0]) & (df.scenarioId == i[1]) & (df.key_num == 8)]))
    #
    # df_missed.user_num = user
    # df_missed.scenario_num = scenario
    # df_missed.nobs = nobs
    # df_missed.ndel = ndel
    # df_missed.ntrue = ntrue
    # df_missed.missedfrac = (df_missed.ntrue + df_missed.ndel - df_missed.nobs)/(df_missed.ntrue + df_missed.ndel)
    # df_missed.to_csv(os.path.join(DATADIR, 'missedfrac.csv'))
    # pdb.set_trace()
    # # require more than 100 keystrokes in a sample for analysis
  #  pdb.set_trace()
    df = groups.filter(lambda x: x['key_num'].count() > 100).reset_index(drop=True)

    df['hloc'], df['vloc'], df['zone'] = utils.keynums_to_keylocs(df.key_num)

    df = utils.get_nloc(df, 2)

    for n in ngraphs:
        df = utils.get_ngraph(df, n)

    #pdb.set_trace()

    df = df[df.tflight2.abs() < 1000.]
    df = df[df.zone.isin([0, 1, 2, 3, 4, 5, 6, 8, 32])].reset_index(drop=True)

    df = utils.bin_samples(df, nbins=bins_per_scen)

    for col in ['user_num', 'scenario_num', 'key_num']:
        assert df[col].dtype in ['int32', 'int64'], \
            "data type of '{}' column must be integer".format(col)

    for col in ['tstart', 'tstop', 'tdwell']:
        assert df[col].dtype in ['float'], \
            "data type of '{}' column must be float".format(col)

    for col in ['key_ascii']:
        assert df[col].dtype in ['object'], \
            "data type of '{}' column must be object".format(col)

    return df


def save_raw(df):
    filename = os.path.join(DATADIR, 'raw_cleaned.csv')
    df.to_csv(filename, index=False)

    return


def read_cleaned():
    filename = os.path.join(DATADIR, 'raw_cleaned.csv')
    df = pd.read_csv(filename, encoding="ISO-8859-1")

    return df



if __name__ == "__main__":
    main()
