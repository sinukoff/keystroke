__all__ = ['driver', 'data_prep', 'trial', 'validate', 'setupfile', 'rvmodel', 'tables', 'coords', 'utils']
import os

ROOTDIR = '/Users/evan/Projects/keystroke/'
DATADIR = '/Users/evan/Projects/keystroke/data/'
#ROOTDIR = '/Users/evan/Code/Insight/keystroke/'
#DATADIR = '/Users/evan/Code/Insight/keystroke/data'
TRIALSDIR = os.path.join(ROOTDIR, 'trials')
RAWDATAFILE = os.path.join(DATADIR, 'keystroke_raw.csv')

assert os.path.exists(ROOTDIR), \
    "ROOTDIR does not exist: \n'{}".format(ROOTDIR)
assert os.path.exists(DATADIR), \
    "DATADIR does not exist: \n'{}".format(DATADIR)
assert os.path.exists(RAWDATAFILE), \
    "RAWDATAFILE does not exist: \n'{}".format(RAWDATAFILE)

if not os.path.exists(TRIALSDIR):
    print("Making directory to store output from different trials")
    os.mkdir(TRIALSDIR)

__version__ = '0.1'
