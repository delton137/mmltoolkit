import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import KFold, ShuffleSplit
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter

#----------------------------------------------------------------------------
def plot_learning_curve(model, X, y, train_sizes=np.linspace(0.2,1,20), fit_exponential=False,
                        ylim=None, cv=KFold(n_splits=5,shuffle=True), title='', include_title=False,
                        units='', n_jobs=-1):
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
    if (include_title): plt.title(title, fontsize=25)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("# training examples", fontsize=22)
    plt.ylabel("Mean absolute error "+units, fontsize=22)
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

    if (fit_exponential):
        def scaling_fn(x, A=1, alpha=1.5):
            return A*np.array(x)**(-1*alpha)

        if (fit_exponential):
            p_fit, p_cov = curve_fit(scaling_fn, train_sizes, test_scores_mean, p0=[1, 1.5])

        x_fit = np.logspace(np.log10(min(train_sizes)), 3.01, 100)
        print(p_fit)

    y_fit = scaling_fn(list(x_fit), A=p_fit[0], alpha=p_fit[1])

    label_string = 'y = %3.2f x^(-%3.2f)' % (p_fit[0], p_fit[1])
    plt.plot(x_fit, y_fit, '--', color="b", label=label_string )

    plt.legend(loc="best", fontsize=22, framealpha=1)
    plt.yscale('log')
    plt.xscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%3.2f"))


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
