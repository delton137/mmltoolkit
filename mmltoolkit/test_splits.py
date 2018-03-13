from sklearn.model_selection import StratifiedShuffleSplit
from mmltoolkit.CV_tools import *
from sklearn.cluster import KMeans


#--------------------------------------------------------------------------------
def print_latex_row(name, scores_dict):
    print("%30s &   %5.2f $\\pm$ %3.2f & %5.2f $\\pm$ %3.2f & %5.2f &  %5.3f &  %5.2f & %5.2f & %5.2f & %5.2f  \\\\" %
                                  (name,
                                        np.mean(scores_dict['train_abs_err']),
                                        np.std(scores_dict['train_abs_err']),
                                        np.mean(scores_dict['test_abs_err']),
                                        np.std(scores_dict['test_abs_err']),
                                        np.mean(scores_dict['test_MAPE']),
                                        np.mean(scores_dict['test_RMSE']),
                                        np.mean(scores_dict['train_R2']),
                                        np.mean(scores_dict['test_R2']),
                                        np.mean(np.sqrt(scores_dict['train_rP'])),
                                        np.mean(np.sqrt(scores_dict['test_rP']))
                                  ))

#--------------------------------------------------------------------------------
def test_splits(X, y, model, groups=None, print_latex=True, n_clusters=10):

    scores_dict_stratificaiton = test_and_plot(X , y,  model,
                            StratifiedShuffleSplit(n_splits=5),
                            groups=groups, plot_title='group-wise stratification')

    scores_dict_shuffle = test_and_plot(X, y, model,
                            ShuffleSplit(n_splits=5),
                            groups=None, plot_title='standard shuffle split')

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    clusters = kmeans.labels_

    scores_dict_clusters = test_and_plot(X, y, model,
                            StratifiedShuffleSplit(n_splits=5),
                            groups=clusters, plot_title='k-means clusters (k=%3i)'%(n_clusters))

    if (print_latex):
        print("\\begin{table*}")
        print("\\begin{tabular}{c c c c c c c c c}")
        print("                name          & MAE$_{\\ff{train}}$   &  MAE$_{\\ff{test}}$  & MAPE$_{\\ff{test}}$ & RMSE$_{\\ff{test}}$  & $R^2_{\\ff{train}}$ &  $R^2 _{\\ff{test}} $&  $r_{\\ff{train}}$ & $r_{\\ff{test}} $         \\\\  ")
        print("\\hline")
        print_latex_row("shuffle", scores_dict_shuffle)
        print_latex_row("stratified", scores_dict_stratificaiton)
        print_latex_row("$k$-means clusters, k = 10", scores_dict_clusters)
        print("\\end{tabular}")
        print("\\end{table*}")
