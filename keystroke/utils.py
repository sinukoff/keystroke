import numpy as np
import pdb
import keystroke
import os
import pandas as pd
DATADIR = keystroke.DATADIR

chars = {}


def keynum_to_ascii(keynums):
    ascii_str = [chr(item) for item in keynums]
    ascii_str = np.array(ascii_str)
    ascii_str[ascii_str == '\r'] = ' '
    #pdb.set_trace()
    return ascii_str



def str_to_words(str):

    return


def key_freq():

    filename = os.path.join(DATADIR, 'raw_cleaned.csv')
    df = pd.read_csv(filename)
    freq = df.groupby(['key_num']).size()
   # print(np.sort(df.key_num.unique()))