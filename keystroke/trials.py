import keystroke
import os
import pickle
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from keystroke import data_prep

TRIALSDIR = os.path.join(keystroke.ROOTDIR, 'trials')


class Trial(object):

    def __init__(self, args, df_features):
        self.keystroke_file = args.infile
        self.min_keys = args.min_keys
        self.n_chunks = args.nchunks
        self.trial_name = args.trialname
        self.feature_cols = args.featurecols
        self.keyboard_zones = args.zones
        self.n_zones_max = args.nzonesmax
        self.med = args.med
        self.mad = args.mad
        self.df_features = df_features
        self.models = {}


    def preprocess_features(self):
        df = self.df_features
        X = impute(df[df.columns[3:]], allow_null=False)
        X = StandardScaler().fit_transform(X)
        self.df_preprocessed = pd.DataFrame(X, index=[df.user_num, df.scenario_num, df.bin_num]).reset_index()


    def add_isoforest(self, args):

        self.models[args.name] = IsolationForest(n_estimators=args.n_estimators,
                                                 max_samples=args.max_samples,
                                                 contamination=args.contamination,
                                                 max_features=args.max_features,
                                                 bootstrap=args.bootstrap,
                                                 n_jobs=args.n_jobs,
                                                 behaviour=args.behaviour,
                                                 random_state=args.random_state,
                                                 verbose=args.verbose
                                                 )
        return


    def to_pickle(self, filepath):
        """
        Save trial object to pickle file.

        Args:
            filepath (string): full path to output file
        """

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


def load(filepath):
    """
    Load posterior object from pickle file.

    Args:
        filename (string): full path to pickle file
    """

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    return obj
