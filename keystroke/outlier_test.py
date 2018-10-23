from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
from keystroke import features, plotting
import pdb

import itertools
import os
import keystroke

bins_per_scen = 8

def main():

    df = features.read_features()
    df = features.preprocess(df).reset_index(drop=True)
    userlist = df.user_num.values
    user_uniq = np.unique(userlist)

    Pthresh = np.linspace(0, 1, bins_per_scen + 1)
    truepos = []
    trueneg = []
    for p in Pthresh:
        truth_labels = np.array([])
        predicted_labels = np.array([])

        for user in user_uniq:
            print(user)
            df_user = df[df.user_num == user]
            df_imposters = df[df.user_num != user]
            results = outlier_test(df_user, df_imposters, p)
            truth_labels = np.append(truth_labels, results[2])
            predicted_labels = np.append(predicted_labels, results[3])

        df_out = pd.DataFrame({'truth': truth_labels, 'pred': predicted_labels})
        df_pos = df_out[df_out.truth == -1]
        df_neg = df_out[df_out.truth == 1]

        npos_success = len(df_pos[df_pos.pred == -1])/float(len(df_pos))
        nneg_success = len(df_neg[df_neg.pred == 1]) / float(len(df_neg))

        truepos.append(npos_success)
        trueneg.append(nneg_success)

    df_roc = pd.DataFrame({'truepos': truepos, 'trueneg': trueneg})

    df_roc.to_csv(os.path.join(keystroke.DATADIR, 'roc_curve_lof.csv'), index=False)

    plotting.plot_roc(trueneg, truepos)


def outlier_test(df_user, df_imposter, Pthresh):

    #lf = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01, shrinking=False)
   # clf = EllipticEnvelope(contamination=0.01)
   # clf = LocalOutlierFactor(novelty=True, n_neighbors=5)
    clf = IsolationForest(n_estimators=100, contamination=0.25, max_features=20, random_state=10, verbose=False)
    user_scenarios = df_user.scenario_num.unique()

    df_user_list = []
    df_imp_list = []

    # One dataframe for each user scenario
    for i in user_scenarios:
        df_user_list.append(df_user[df_user.scenario_num == i].set_index(['user_num', 'scenario_num', 'bin_num']))

    Nscen = len(df_user_list)

    Pright_user = []
    Pwrong_user = []
    success_neg = []
    success_pos = []

    truth_label = []
    predicted_label = []

    for i, j in itertools.combinations(range(Nscen), 2):
        df1 = df_user_list[i]
        df2 = df_user_list[j]

        user_fit = train_mod(clf, df1.values)
        diffscen_pred = predict_mod(user_fit, df2.values)
       # pdb.set_trace()

        Nright = len(diffscen_pred[diffscen_pred == 1])
        Nwrong = len(diffscen_pred[diffscen_pred == -1])
        Pright = Nright / float(Nright + Nwrong)
        Pwrong = Nwrong / float(Nright + Nwrong)
        Pright_user.append(Pright)
        Pwrong_user.append(Pwrong)
        if Pright >= Pthresh:
            success_neg.append(1)
            predicted_label.append(1)
        else:
            success_neg.append(0)
            predicted_label.append(-1)
        truth_label.append(1)

    success_neg_frac = np.sum(success_neg) / float(len(success_neg))

    df_imp = df_imposter.set_index(['user_num', 'scenario_num', 'bin_num'])

    for i in df_imposter.groupby(['user_num', 'scenario_num']).groups.keys():
        df_imp_list.append(df_imp.loc[i])

    for i in range(Nscen):
        df1 = df_user_list[i]
        user_fit = train_mod(clf, df1.values)
        for df_imp_i in df_imp_list:
            imp_pred = predict_mod(user_fit, df_imp_i.values)
            Nright = len(imp_pred[imp_pred == -1])
            Nwrong = len(imp_pred[imp_pred ==  1])
            Pright = Nright / float(Nright + Nwrong)
            Pwrong = Nwrong / float(Nright + Nwrong)
            if Pright > (1.0 - Pthresh):
                success_pos.append(1)
                predicted_label.append(-1)
            else:
                success_pos.append(0)
                predicted_label.append(1)
            truth_label.append(-1)

    success_pos_frac = np.sum(success_pos)/float(len(success_pos))

    return success_neg_frac, success_pos_frac, truth_label, predicted_label



def train_mod(clf, X_train):

    return clf.fit(X_train)

def predict_mod(clf, X_test):

        return clf.predict(X_test)



if __name__ == "__main__":
    main()
