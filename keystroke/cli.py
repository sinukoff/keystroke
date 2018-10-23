"""
Command Line Interface
"""
import os
from argparse import ArgumentParser
import warnings

import keystroke.driver

warnings.simplefilter("ignore")
warnings.simplefilter('once', DeprecationWarning)


def main():
    psr = ArgumentParser(
        description="KeyID: Identifying online users by their keystrokes", prog='KeyID'
    )
    psr.add_argument('--version',
                     action='version',
                     version="%(prog)s {}".format(keystroke.__version__),
                     help="Print version number and exit."
                     )

    subpsr = psr.add_subparsers(title="subcommands", dest='subcommand')

    # In the parent parser, we define arguments and options common to
    # all subcommands.
    psr_parent = ArgumentParser(add_help=False)
    psr_parent.add_argument(
        '-t', '--trialname', dest='trialname', type=str,
        default='test',
        help="Name of trial directory"
    )

    # Add columns to the raw keystroke dataset
    psr_update_raw = subpsr.add_parser(
        'update_raw', parents=[psr_parent],
        description="Add columns to the raw keystroke dataset"
    )
    psr_update_raw.add_argument('-n', '--ngraph_max', dest='ngraph_max',
                             action='store', default=2, type=int,
                             help="Flight times and dwell times will be computed for sequences of size 1 to n"
                             )
    psr_update_raw.add_argument('-o', '--outfile', dest='outfile',
                                action='store', default='keystroke_updated.csv', type=str,
                                help="Name of file to be saved"
                                )
    psr_update_raw.set_defaults(func=keystroke.driver.update_raw)

    psr_trial = subpsr.add_parser(
        'trial', parents=[psr_parent],
        description="Make a new trial"
    )

    psr_trial.add_argument('-i', '--infile', action='store', type=str,
                           default='keystroke_updated.csv',
                           help="Name of file in which feature set will be stored"
                           )
    psr_trial.add_argument('-min_keys', dest='min_keys',
                           action='store', default=100, type=int,
                           help="Responses with fewer than this number of keystrokes will be excluded."
                           )

    psr_trial.add_argument('-c', '--chunks', dest='nchunks',
                           action='store', default=8, type=int,
                           help="Partition each response into this many equal-sized chunks"
                           )
    psr_trial.add_argument('-tmax', dest='max_flight',
                           action='store', default=1000, type=int,
                           help="Maximum flight time in milliseconds. Longer flight times will be excluded."
                           )
    psr_trial.add_argument('-z', '--zones', nargs='+', type=int,
                           default=[1, 2, 3, 4, 5, 6, 32, 8],
                           help="List of keyboard zones to include"
                           )
    psr_trial.add_argument('-nzonesmax', '--nzonesmax', dest='nzonesmax', type=int,
                           default=20,
                           help="Features will be computed from the nzonesmax most common zone transitions",
                           )
    psr_trial.add_argument('-f', '--featurecols', dest='featurecols', nargs='+', type=str,
                           default=['tdwell1', 'tdwell2', 'tflight2'],
                           help="List of feature types e.g tdwell1 tdwell2 tflight2",
                           )
    psr_trial.add_argument('-med', '--med', action='store_true',
                           default=True,
                           help="Include medians in feature set"
                           )
    psr_trial.add_argument('-mad', '--mad', action='store_true',
                           default=False,
                           help="Include median absolute deviations in feature set"
                           )
    psr_trial.add_argument('-o', '--outfile', action='store', type=str,
                           default='features.csv',
                           help="Name of file in which feature set will be stored"
                           )


    psr_trial.set_defaults(func=keystroke.driver.make_trial)



    psr_tsne = subpsr.add_parser(
         'tsne', parents=[psr_parent],
         description="Plot t-SNE visualization, points colored by user"
     )

    psr_tsne.add_argument('-i', '--infile', dest='infile',
                          action='store', default='features.csv', type=str,
                          help="Name of file containing feature dataset"
                          )

    psr_tsne.add_argument('-o', '--outfile', dest='outfile',
                                action='store', default='plot_tsne.pdf', type=str,
                                help="Name of file to be saved"
                          )

    psr_tsne.add_argument('-p', '--perplexity', dest='perplexity',
                                action='store', default=5, type=int,
                                help="t-SNE perplexity parameter"
                          )

    psr_tsne.add_argument('-l', '--learnrate', dest='learnrate',
                                action='store', default=200, type=int,
                                help="t-SNE learning rate parameter"
                          )


    psr_tsne.set_defaults(func=keystroke.driver.tsne)


    psr_isoforest = subpsr.add_parser(
        'add_isoforest', parents=[psr_parent],
        description="Add isolation forest model"
    )
    psr_isoforest.add_argument('-name', '--name', dest='name',
                          action='store', default='isoforest1', type=str,
                          help="name to assign model"
                          )
    psr_isoforest.add_argument('-n_estimators', '--n_estimators', dest='n_estimators',
                          action='store', default=100, type=int,
                          help="sklearn Isolation Forest n_estimators parameter "
                          )
    psr_isoforest.add_argument('-max_samples', '--max_samples', dest='max_samples',
                          action='store', default=1, type=int,
                          help="sklearn Isolation Forest max_samples parameter "
                          )
    psr_isoforest.add_argument('-contamination', '--contamination', dest='contamination',
                          action='store', default=0.1, type=float,
                          help="sklearn Isolation Forest contamination parameter "
                          )
    psr_isoforest.add_argument('-max_features', '--max_features', dest='max_features',
                          action='store', default=1, type=float,
                          help="sklearn Isolation Forest max_features parameter"
                          )
    psr_isoforest.add_argument('-bootstrap', '--bootstrap', dest='bootstrap',
                          action='store_true', default=False,
                          help="sklearn Isolation Forest bootstrap parameter"
                          )
    psr_isoforest.add_argument('-n_jobs', '--n_jobs', dest='n_jobs',
                          action='store', default=1, type=int,
                          help="sklearn Isolation Forest n_jobs parameter "
                          )
    psr_isoforest.add_argument('-behaviour', '--behaviour', dest='behaviour',
                          action='store', type=str, choices=['old', 'new'],
                          default='old',
                          help="sklearn Isolation Forest behaviour parameter"
                          )
    psr_isoforest.add_argument('-random_state', '--random_state', dest='random_state',
                          action='store', default=1, type=int,
                          help="sklearn Isolation Forest random_state parameter "
                          )
    psr_isoforest.add_argument('-verbose', '--verbose', dest='verbose',
                          action='store_true', default=False,
                          help="sklearn Isolation Forest verbose parameter"
                          )

    psr_isoforest.set_defaults(func=keystroke.driver.add_isoforest)

    args = psr.parse_args()
    args.func(args)

   # # Add columns to the raw keystroke dataset
   #  psr_validate = subpsr.add_parser(
   #      'validate', parents=[psr_parent],
   #      description="Validate outlier rejection model"
   #  )

if __name__ == '__main__':
    main()


#     # Plotting
#     psr_plot = subpsr.add_parser('plot', parents=[psr_parent], )
#     psr_plot.add_argument('-t', '--type',
#                           type=str, nargs='+',
#                           choices=['rv', 'corner', 'trend', 'derived'],
#                           help="type of plot(s) to generate"
#                           )
#
#     psr_plot.add_argument('--gp',
#                           dest='gp',
#                           action='store_true',
#                           default=False,
#                           help="Make a multipanel plot with GP bands. For use only with GPLikleihood objects"
#                           )
#
#     psr_plot.set_defaults(func=radvel.driver.plots)
#
#     # MCMC
#     psr_mcmc = subpsr.add_parser(
#         'mcmc', parents=[psr_parent],
#         description="Perform MCMC exploration"
#     )
#     psr_mcmc.add_argument(
#         '--nsteps', dest='nsteps', action='store', default=10000, type=float,
#         help='Number of steps per chain [10000]', )
#     psr_mcmc.add_argument(
#         '--nwalkers', dest='nwalkers', action='store', default=50, type=int,
#         help='Number of walkers. [50]',
#     )
#     psr_mcmc.add_argument(
#         '--nensembles', dest='ensembles', action='store', default=8, type=int,
#         help="Number of ensembles. Will be run in parallel on separate CPUs [8]"
#     )
#     psr_mcmc.add_argument(
#         '--maxGR', dest='maxGR', action='store', default=1.01, type=float,
#         help="Maximum G-R statistic for chains to be deemed well-mixed and halt the MCMC run [1.01]"
#     )
#     psr_mcmc.add_argument(
#         '--burnGR', dest='burnGR', action='store', default=1.03, type=float,
#         help="Maximum G-R statistic to stop burn-in period [1.03]"
#     )
#     psr_mcmc.add_argument(
#         '--minTz', dest='minTz', action='store', default=1000, type=int,
#         help="Minimum Tz to consider well-mixed [1000]"
#     )
#     psr_mcmc.add_argument(
#         '--minsteps', dest='minsteps', action='store', default=1000, type=int,
#         help="Minimum number of steps per walker before convergence tests are performed [1000]"
#     )
#     psr_mcmc.add_argument(
#         '--thin', dest='thin', action='store', default=1, type=int,
#         help="Save one sample every N steps [default=1, save all samples]"
#     )
#     psr_mcmc.add_argument(
#         '--serial', dest='serial', action='store', default=False, type=bool,
#         help='''\
# If True, run MCMC in serial instead of parallel. [False]
# '''
#     )
#     psr_mcmc.set_defaults(func=radvel.driver.mcmc)
#
#     # Derive physical parameters
#     psr_physical = subpsr.add_parser(
#         'derive', parents=[psr_parent],
#         description="Multiply MCMC chains by physical parameters. MCMC must"
#                     + "be run first"
#     )
#
#     psr_physical.set_defaults(func=radvel.driver.derive)
#
#     # Information Criteria comparison (BIC/AIC)
#     psr_ic = subpsr.add_parser('ic', parents=[psr_parent], )
#     psr_ic.add_argument('-t',
#                         '--type', type=str, nargs='+', default='trend',
#                         choices=['nplanets', 'e', 'trend', 'jit', 'gp'],
#                         help="parameters to include in BIC/AIC model comparison"
#                         )
#
#     psr_ic.add_argument('-m',
#                         '--mixed', dest='mixed', action='store_true',
#                         help="flag to compare all models with the fixed parameters mixed and matched rather than" \
#                              + " treating each model comparison separately. This is the default. " \
#                         )
#     psr_ic.add_argument('-u',
#                         '--un-mixed', dest='mixed', action='store_false',
#                         help="flag to treat each model comparison separately (without mixing them) " \
#                              + "rather than comparing all models with the fixed parameters mixed and matched."
#                         )
#     psr_ic.add_argument('-f',
#                         '--fixjitter', dest='fixjitter', action='store_true',
#                         help="flag to fix the stellar jitters at the nominal model best-fit value"
#                         )
#     psr_ic.add_argument('-n',
#                         '--no-fixjitter', dest='fixjitter', action='store_false',
#                         help="flag to let the stellar jitters float during model comparisons (default)"
#                         )
#     psr_ic.add_argument('-v',
#                         '--verbose', dest='verbose', action='store_true',
#                         help="Print some more detail"
#                         )
#     psr_ic.set_defaults(func=radvel.driver.ic_compare, fixjitter=False, unmixed=False, \
#                         mixed=True)
#
#     # Tables
#     psr_table = subpsr.add_parser('table', parents=[psr_parent], )
#     psr_table.add_argument('-t', '--type',
#                            type=str, nargs='+',
#                            choices=['params', 'priors', 'rv', 'ic_compare'],
#                            help="type of plot(s) to generate"
#                            )
#     psr_table.add_argument(
#         '--header', action='store_true',
#         help="include latex column header. Default just prints data rows"
#     )
#     psr_table.add_argument('--name_in_title',
#                            dest='name_in_title',
#                            action='store_true',
#                            default=False,
#                            help='''
#         Include star name in table headers. Default just prints
#         descriptive titles without star name [False]
#     '''
#                            )
#
#     psr_table.set_defaults(func=radvel.driver.tables)
#
#     # Report
#     psr_report = subpsr.add_parser(
#         'report', parents=[psr_parent],
#         description="Merge output tables and plots into LaTeX report"
#     )
#     psr_report.add_argument(
#         '--comptype', dest='comptype', action='store',
#         default='ic', type=str,
#         help='Type of model comparison table to include. \
#         Default: ic')
#
#     psr_report.add_argument(
#         '--latex-compiler', default='pdflatex', type=str,
#         help='Path to latex compiler'
#     )
#
#     psr_report.set_defaults(func=radvel.driver.report)


    # if args.outputdir is None:
    #     setupfile = args.setupfn
    #     print(setupfile)
    #     system_name = os.path.basename(setupfile).split('.')[0]
    #     outdir = os.path.join('./', system_name)
    #     args.outputdir = outdir
    #
    # if not os.path.isdir(args.outputdir):
    #     os.mkdir(args.outputdir)



