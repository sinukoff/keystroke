import os
import numpy as np
import pandas as pd
import keystroke
from keystroke import utils

DATADIR = keystroke.DATADIR

def main():
    """
    Main executable to read, clean, and save raw keystroke data
    """
    df = read_raw()
    df = clean_raw(df)
    save_raw(df)

    return



def addcols_raw(df, ngraph_max):
    """"
    -Modifies column names of raw keystroke data and adds new columns:
        -for N = 0 to ngraph_max - 1, the following columns are added:
            tdwellN: Sum of dwell times for each key and N previous keys
            tflightN: Sum of flight times between each key and N previous keys (meaningless for N=0)
            zoneN:, Zone of key N rows before current key

    Args:
        df (DataFrame): Raw keystroke time-series data.
        ngraphs_max (int): Flight times and dwell times will be computed for sequences of size 1 to n
    """

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
    df['tdwell'] = df.tstop - df.tstart

    df.sort_values(by=['user_num', 'tstart'], inplace=True)
    # Remove duplicate keystrokes
    df.drop_duplicates(['user_num', 'tstart', 'tstop'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ['user_num', 'scenario_num', 'key_num']:
        assert df[col].dtype in ['int32', 'int64'], \
            "data type of '{}' column must be integer".format(col)

    for col in ['tstart', 'tstop', 'tdwell']:
        assert df[col].dtype in ['float'], \
            "data type of '{}' column must be float".format(col)

    for col in ['key_ascii']:
        assert df[col].dtype in ['object'], \
            "data type of '{}' column must be object".format(col)

    # Add columns specifying location on keyboard
    df['hloc'], df['vloc'], df['zone'] = utils.keynums_to_keylocs(df.key_num)

    for n in np.arange(2, ngraph_max + 1, 1):
        df = utils.get_nloc(df, n)

    for n in np.arange(1, ngraph_max + 1, 1):
        df = utils.get_ngraph(df, n)

    return df


def filter_raw(df, min_keys, min_flight, zones):
    """
    -Removes responses with fewer than min_keys keystrokes
        -Adds columns with dwell times flight times:
            -tdwellN = Sum of dwell times for current keystroke and previous N-1 keystrokes.
            -tflightN = Sum of flight times for current keystroke and previous N-1 keystrokes.
    """

    groups = df.groupby(['user_num', 'scenario_num'])

    # Require more than 100 keystrokes in a sample for analysis
    df = groups.filter(lambda x: x['key_num'].count() > min_keys).reset_index(drop=True)
    df = df[df.tflight2.abs() < min_flight]
    df = df[df.zone.isin(zones)].reset_index(drop=True)

    return df



def read_cleaned():
    filename = os.path.join(DATADIR, 'raw_cleaned.csv')
    df = pd.read_csv(filename, encoding="ISO-8859-1")

    return df


if __name__ == "__main__":
    main()
