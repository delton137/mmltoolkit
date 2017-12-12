import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import KFold, ShuffleSplit


#----------------------------------------------------------------------------
def plot_learning_curve(model, X, y, train_sizes=np.linspace(0.2,1,20),
                        ylim=None, cv=KFold(n_splits=5,shuffle=True), title='', n_jobs=-1):
    """
    Generate a simple plot of the test and training learning curve.
    Adapted from code from scikitlearn.org

    Required args:
        model : ML model (has "fit" and "predict" methods)
        X : design/data matrix, array-like, shape (n_samples, n_features)
        y : target vector, array-like, shape (n_samples) or (n_samples, n_features)
    Optional args:
        ylim : tuple, shape (ymin, ymax), Defines minimum and maximum yvalues plotted.
        cv : int, cross-validation generator or an iterable, optional
        n_jobs : integer, optional, Number of jobs to run in parallel (default -1), -1 for automatic parallelization
    """

    plt.figure(figsize=(10,9))
    plt.clf()
    #plt.title(title, fontsize=25)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("# training examples", fontsize=22)
    plt.ylabel("Mean absolute error (kJ/cc)", fontsize=22)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_absolute_error')
    train_scores = -1*train_scores
    test_scores = -1*test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    test_scores_max = np.max(test_scores, axis=1)
    test_scores_min = np.min(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    #plt.fill_between(train_sizes, test_scores_min,
    #                 test_scores_max, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="test score")

    plt.legend(loc="best", fontsize=16)
    plt.yscale('log')
    plt.xscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    #ax.loglog()

    #ax.set_xticklabels(x_ticks, rotation=0, fontsize=10)
    #ax.set_yticklabels(y_ticks, rotation=0, fontsize=10)
    #plt.rcParams['xtick.labelsize']=8
    #plt.rcParams['ytick.labelsize']=8
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.tick_params(length=7.0)
    plt.tick_params(which='minor', length=5.0)

    return plt
