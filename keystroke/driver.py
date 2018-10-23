import os
import pandas as pd
import keystroke
from keystroke import data_prep, features, utils, plotting, trials
import pdb


def update_raw(args):
    df = pd.read_csv(keystroke.RAWDATAFILE)
    df = data_prep.addcols_raw(df, args.ngraph_max)
    saveto = os.path.join(keystroke.DATADIR, args.outfile)
    df.to_csv(saveto, index=False)


def make_trial(args):

    infile = os.path.join(keystroke.DATADIR, args.infile)
    print("Reading file: {}".format(infile))
    df_keystroke = pd.read_csv(infile)
    df_keystroke = data_prep.filter_raw(df_keystroke, args.min_keys, args.max_flight, args.zones)
   # print(len(df_keystroke))
    df_keystroke = utils.bin_samples(df_keystroke, nbins=args.nchunks)
    #print(len(df_keystroke))
    df_keystroke = features.keystroke_missedfrac_trim(df_keystroke)
    df_features = features.keystrokes_to_features(df_keystroke, args.zones, args.nzonesmax, args.featurecols, args.med, args.mad)

    trialdir = os.path.join(keystroke.TRIALSDIR, args.trialname)
    if not os.path.exists(trialdir):
        os.mkdir(trialdir)

    saveto_csv = os.path.join(trialdir, args.outfile)
    df_features.to_csv(saveto_csv, index=False)
    trial = trials.Trial(args, df_features)
    trial.preprocess_features()

    saveto_pkl = os.path.join(trialdir, 'trial.pkl')
    trial.to_pickle(saveto_pkl)


def tsne(args):
    trialdir = os.path.join(keystroke.TRIALSDIR, args.trialname)
    infile = os.path.join(trialdir, args.infile)
    df = pd.read_csv(infile)
    df = features.preprocess(df)
    X = df[df.columns[3:]]
    y = df.user_num.values
    nchunks = int(df.bin_num.max())
    saveto = os.path.join(trialdir, args.outfile)
    plotting.plot_tsne(X, y, nchunks=nchunks, perplexity=args.perplexity, learnrate=args.learnrate, saveto=saveto)


def add_isoforest(args):
    trialdir = os.path.join(keystroke.TRIALSDIR, args.trialname)
    trialfile = os.path.join(trialdir, 'trial.pkl')
    trial = trials.load(trialfile)
    trial.add_isoforest(args)
    saveto_pkl = os.path.join(trialdir, 'trial.pkl')
    trial.to_pickle(saveto_pkl)






