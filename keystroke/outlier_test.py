from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from keystroke import features, plotting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeavePOut

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import itertools

import random

import pdb
bins_per_scen = 8

def main():

    df = features.read_features()
  #  pdb.set_trace()
    df = features.preprocess(df).reset_index(drop=True)
    #pdb.set_trace()
    results = {}
   # userlist = df.index.get_level_values(level=0).values
    userlist = df.user_num.values
    #user_uniq = np.unique(userlist)
    user_uniq = np.unique(userlist)


    Pstep = 1./bins_per_scen
    Pthresh = np.linspace(0, 1, bins_per_scen+1)
    truepos = []
    trueneg = []
    for p in Pthresh:
        tn = []
        tp = []
        truth_labels = np.array([])
        predicted_labels = np.array([])

        for user in user_uniq:
            df_user = df[df.user_num == user]
            df_imposters = df[df.user_num != user]
            results = outlier_test(df_user, df_imposters, p)
            tn.append(results[0])
            tp.append(results[1])
            truth_labels = np.append(truth_labels, results[2])
            predicted_labels = np.append(predicted_labels, results[3])

        df_out = pd.DataFrame({'truth': truth_labels, 'pred': predicted_labels})
        df_pos = df_out[df_out.truth == -1]
        df_neg = df_out[df_out.truth == 1]

        npos_success = len(df_pos[df_pos.pred == -1])/float(len(df_pos))
        nneg_success = len(df_neg[df_neg.pred == 1]) / float(len(df_neg))

        truepos.append(npos_success)
        trueneg.append(nneg_success)

    pdb.set_trace()

    df_roc = pd.DataFrame({'truepos': truepos, 'trueneg': trueneg})





    print(confusion_matrix(predicted_labels, truth_labels))
    print(classification_report(predicted_labels, truth_labels))



    print(np.mean(tn), np.mean(tp))

    pdb.set_trace()
    #y = df.user_num.values
   # X = df.drop(['user_num', 'scenario_num', 'bin_num'])

    xtrain, ytrain, xtest, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svm = OneClassSVM()

    param_grid = {'n_neighbors': np.arange(0, 5, 0.1)}

    svm_cv = GridSearchCV(svm, param_grid, cv=5)
    svm_cv.fit(X, y)

    svm_cv.best_params_
    svm_cv.best_score_


    roc_auc_scores = cross_val_score(svm, X, y, cv=5, scoring='roc_auc')
    fpr, tpr, thresholds = roc_curve(ytest, ypred_prob)
    roc_auc_score = roc_auc_curve(ytest, ypred_prob)



def outlier_test(df_user, df_imposter, Pthresh):


  #  df_user = df_user.set_index(['user_num', 'scenario_num', 'bin_num'])
  #  df_imposter = df_imposter.set_index(['user_num', 'scenario_num', 'bin_num'])

   # X_user = df_user.values
   # y_user = df_user.index.get_level_values(level=0)
   # clf = OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
   # clf = EllipticEnvelope(contamination=0.01)
    clf = IsolationForest(n_estimators=100, contamination=0.25, max_features=20, random_state=10, verbose=False)
    lpo = LeavePOut(1)
   # X_user = lpo.split(X_user)

   # X_imp = df_imposter.values
   # y_imp = df_imposter.index.get_level_values(level=0)

  #  df_user = df_user.reset_index()
  #  df_imposter = df_imposter.reset_index()

    user_scenarios = df_user.scenario_num.unique()
    #unique scenarios for each user
   # samples_user = df_user.drop_duplicates(['user_num', 'scenario_num'])[['user_num', 'scenario_num']].values
   # Nscen = len(samples_user)/ Nchunks

    # unique imposters
    # samples_imp = df_imposter.drop_duplicates(['user_num', 'scenario_num'])[['user_num', 'scenario_num']].values
    # Nimp = len(samples_imp)

    df_user_list = []
    df_imp_list = []

    #One dataframe for each user scenario
    for i in user_scenarios:
        df_user_list.append(df_user[df_user.scenario_num == i].set_index(['user_num', 'scenario_num', 'bin_num']))

    #USE THIS IF TRAINING EACH USER ON CLUSTER OF ALL SCENARIOS FOR THAT USER
   # for i in user_scenarios:
   #     df_user_list.append(df_user.set_index(['user_num', 'scenario_num', 'bin_num']))

    Nchunks = int(np.max(df_user.bin_num))
    inds_user = lpo.split(range(Nchunks))


    imp_groups = df_imposter.groupby(['user_num', 'scenario_num']).groups


    imp_inds = [df_imposter.groupby(['user_num', 'scenario_num']).groups[i].values for i in imp_groups.keys()]
   # pdb.set_trace()
        #imp_array =
    #for i in imp_groups:


# df_imposter = df_imposter.set_index(['user_num', 'scenario_num', 'bin_num'])
    ncrit = 3

    Nscen = len(df_user_list)

    Pright_user = []
    Pwrong_user = []
   # Pthresh = 0.2 #(need this fraction matching)
    success_neg = []
    success_pos = []

    success_neg_frac_user = []
    success_pos_frac_user = []
    truth_label = []
    predicted_label = []

    for i, j in itertools.combinations(range(Nscen), 2):
        df1 = df_user_list[i]
        df2 = df_user_list[j]

        user_fit = train_svm(clf, df1.values)
       # samescen_pred = predict_svm(clf, df1.values)
       # print(samescen_pred)
        diffscen_pred = predict_svm(user_fit, df2.values)
       # print(diffscen_pred)
        Nright = len(diffscen_pred[diffscen_pred == 1])
        Nwrong = len(diffscen_pred[diffscen_pred == -1])
        Pright = Nright / float(Nright + Nwrong)
        Pwrong = Nwrong / float(Nright + Nwrong)
        Pright_user.append(Pright)
       # print(Pright)
        Pwrong_user.append(Pwrong)
        if Pright >= Pthresh:
            success_neg.append(1)
            predicted_label.append(1)
        else:
            success_neg.append(0)
            predicted_label.append(-1)
        truth_label.append(1)

    success_neg_frac = np.sum(success_neg) / float(len(success_neg))
   # success_neg_frac_user.append(success_neg_frac)
   # pdb.set_trace()
    #for i in df_imposter.drop

    df_imp = df_imposter.set_index(['user_num', 'scenario_num', 'bin_num'])

    for i in df_imposter.groupby(['user_num', 'scenario_num']).groups.keys():
        df_imp_list.append(df_imp.loc[i])

   # df_imp_list.append(df_imposter[df_imposter.scenario_num == i].set_index(['user_num', 'scenario_num', 'bin_num']))
   # df2 = df_imposter.set_index(['user_num', 'scenario_num', 'bin_num']).values

    for i in range(Nscen):
        df1 = df_user_list[i]
        user_fit = train_svm(clf, df1.values)
        for df_imp_i in df_imp_list:
            imp_pred = predict_svm(clf, df_imp_i.values)
           # print(imp_pred)
            Nright = len(imp_pred[imp_pred == -1])
            Nwrong = len(imp_pred[imp_pred ==  1])
            Pright = Nright / float(Nright + Nwrong)
            Pwrong = Nwrong / float(Nright + Nwrong)
           # print(Pright)
            #Pright_user.append(Pright)
           # Pwrong_user.append(Pwrong)
            if Pright > (1.0 - Pthresh):
                success_pos.append(1)
                predicted_label.append(-1)
            else:
                success_pos.append(0)
                predicted_label.append(1)
            truth_label.append(-1)

    success_pos_frac = np.sum(success_pos)/float(len(success_pos))
    #success_pos_frac_user.append(success_pos_frac)
   # print(success_neg_frac, success_pos_frac)
    #pdb.set_trace()
    return success_neg_frac, success_pos_frac, truth_label, predicted_label

  # for dfi in df_user_list:
  #      # print(dfi.index)
  #      # imp_ids = random.choices(samples_imp, k=Nsamp_imp)
  #      # pdb.set_trace()
  #       user_pred_success = np.array([])
  #       trueposrate_imps = []
  #       success_trueneg = np.array([])
  #       for ind_user_in, ind_user_out in inds_user:
  #           user_fit = train_svm(clf, dfi.iloc[ind_user_in].values)
  #           user_pred = predict_svm(user_fit, dfi.iloc[ind_user_out].values)
  #           #print(clf.predict(dfi.iloc[ind_user_in].values), user_pred)
  #          # print(len(imp_pred[imp_pred == 1]), len(imp_pred))
  #           user_pred_success = np.append(user_pred_success, user_pred[0])
  #          # imp_pred_success = np.append(imp_pred_success, imp_pred[0])
  #
  #           #For that user, what is prob of imposter
  #           #imp_pred_success = np.array([])
  #           success_truepos = []
  #           for i in imp_inds:
  #               imp_pred = predict_svm(user_fit, df_imposter.loc[i].values)
  #               if len(imp_pred[imp_pred == 1]) <= ncrit:
  #                   success_truepos.append(1)
  #               else:
  #                   success_truepos.append(0)
  #
  #               #imp_pred_success = np.append(imp_pred_success, len(imp_pred[imp_pred == -1]))
  #               #trueposrate_imps stores fraction of imposters identified as imposters for that LeaveOneOut
  #               trueposrate_imps.append(np.sum(success_truepos)/float(len(success_truepos)))
  #
  #       # trueposrate_user stores fraction of imposters identified as imposters for that user
  #       trueposrate_user = np.mean(trueposrate_imps)
  #
  #           #imp_pred_success = np.mean(imp_pred_success)
  #
  #       if len(user_pred_success[user_pred_success == 1] > ncrit):
  #           success_trueneg = np.append(success_trueneg, 1)
  #       else:
  #           success_trueneg = np.append(success_trueneg, 0)
  #
  #   pdb.set_trace()
  #
  #   return success_trueneg, trueposrate_user

    #df_imp =

    #random numbers to select imposters
 #   Nimp = len(df.imposter)
   # Nsamp = 10
   # ind_imps = np.randint(0, Nimp, Nsamp)

   # for i in
      #  ind_imposter =
      #  ind_imp =
   #     result1 = run_svm(df_user[ind_user_in], df_user[ind_user_out], df_user[ind_imposter])


   # for ind_imp in np.randint(0, Nimp, Nsamp):
   #     result3 = run_svm(ind_imp)



def train_svm(clf, X_train):

    return clf.fit(X_train)

def predict_svm(clf, X_test):
        return clf.predict(X_test)

  #  y_pred_train = clf.predict(X_train)
  #  y_pred_test = clf.predict(X_test)
  #  y_pred_outliers = clf.predict(X_outliers)
   # n_error_train = y_pred_train[y_pred_train == -1].size
   # n_error_test = y_pred_test[y_pred_test == -1].size
   # n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


#clf = OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)

if __name__ == "__main__":
    main()
