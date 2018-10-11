from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def pca_lda():

    LDA()
    clf = make_pipeline(PCA(), LDA()) #LDA can overfit if PCA not done first
    scores = cross_val_score(clf, cv=5)
