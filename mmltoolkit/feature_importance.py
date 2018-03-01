
from sklearn.linear_model import Ridge, Lasso, LinearRegression, RandomizedLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
from mmltoolkit.CV_tools import grid_search
import numpy as np

#------------------------------------------------------------------------------------------
def fi_print_latex(sorted_feature_names, sorted_coeffs, name="", num_to_print='all'):
    """
        helper function to print a LaTeX table of feature importances in a pretty format
        args:
            sorted_feature_names : list of feature names
            sorted_coeffs : list of importance values
    """

    if (num_to_print == 'all'):
        num_to_print = len(sorted_feature_names)
    else:
        num_to_print = num_to_print

    print("\\begin{table}")
    print("\\begin{tabular}{c c}")
    print("feature &  coeff. \\\\")
    for i in range(num_to_print):
        if ( (sorted_coeffs[i] > 0) & (min(sorted_coeffs) < 0) ):
            print("%10s & +%5.3f \\\\" % (sorted_feature_names[i], sorted_coeffs[i]))
        elif (sorted_coeffs[i] == 0.0):
            print("%10s &  %5.3f \\\\" % (sorted_feature_names[i], abs(sorted_coeffs[i])))
        else:
            print("%10s & %5.3f \\\\" % (sorted_feature_names[i], sorted_coeffs[i]))
    print("\\end{tabular}")
    print("\\caption{"+name+" feature importances.}")
    print("\\end{table}")


#------------------------------------------------------------------------------------------
def mean_decrease_accuracy(x, y, feature_names, model=RandomForestRegressor(n_estimators = 50),
                          num_to_print='all', print_latex=True):
    """
        Feature importance by mean decrease accuracy.
        Proposed for random forrests in Breiman, L. Machine Learning (2001) 45: 5. https://doi.org/10.1023/A:1010933404324
        Measures decrease in accuracy values when a given feature is randomly permuted in the dataset.
         If the decrease is low, then the feature is not important, and vice-versa.
        The importances are reported as fractional (percentage) decreases in accuracy after permutation
        Required arguments:
            x : data matrix (number samples x number of features), NumPy array
            y : target property vector, as NumPy array
            feature_names : list of feature names
        Optional arguments:
            model : a scikit-learn model, if none is supplied random forest is used.
        returns:
            sorted_feature_names : sorted list of feature names, most to least important
            sorted_importances : the importances  sorted by absolute value
    """
    from sklearn.cross_validation import ShuffleSplit
    from sklearn.metrics import r2_score
    from collections import defaultdict

    mdoel = model.fit(x, y)

    scores = defaultdict(list)

    #crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
        X_train, X_test = x[train_idx], x[test_idx]
        Y_train, Y_test = y[train_idx], y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    print( sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True) )

    #sort dictionary
    sorted_feature_names = sorted(f_imps, key=f_imps.__getitem__, reverse=True)
    sorted_values = sorted(f_imps.values(), reverse=True)

    if (print_latex): fi_print_latex(sorted_feature_names, sorted_values, name="random forest mean decrease impurity")

    return sorted_feature_names, sorted_values

#------------------------------------------------------------------------------------------
def random_forest_feature_importance(x, y, feature_names, num_to_print='all', print_latex=False, n_estimators=100):
    """ Calculation of feature importance scikit-learn random forest regression feature importance
        In the context of regression, this is done by looking at the variance.
        In the context of classification, this is typically done with the Gini impurity measurement
        REFS:
            Gilles Louppe, et al. "Understanding variable importances in forests of randomized trees"
        Required arguments:
            x : data matrix (number samples x number of features), NumPy array
            y : target property vector, as NumPy array
            feature_names : list of feature names
        returns:
            sorted_feature_names : sorted list of feature names, most to least important
            sorted_importances  : the importances sorted by absolute value
    """
    #rf = grid_search(x, y, RandomForestRegressor(), param_grid={"n_estimators": np.linspace(10, 120, 12).astype('int')}, verbose=False)
    rf =   RandomForestRegressor( n_estimators= n_estimators ) #we want to ues a lot of regressors !
    rf = rf.fit(x, y)

    num_features = len(feature_names)

    #create dictionary
    f_imps = {}
    for i in range(num_features):
        f_imps[feature_names[i]] = rf.feature_importances_[i]

    #sort dictionary
    sorted_feature_names = sorted(f_imps, key=f_imps.__getitem__, reverse=True)
    sorted_values = sorted(f_imps.values(), reverse=True)

    if (print_latex): fi_print_latex(sorted_feature_names, sorted_values, name="random forest mean decrease impurity")

    return sorted_feature_names, sorted_values




#------------------------------------------------------------------------------------------
def LASSO_feature_importance(x, y, feature_names, print_latex=False, num_to_print='all', alpha=.01):
    """ Calculation of feature importance using coefficients in LASSO model.
        "least absolute shrinkage and selection operator"
        Required arguments:
            x : data matrix (number samples x number of features), NumPy array
            y : target property vector, as NumPy array
            feature_names : list of feature names
        returns:
            sorted_feature_names : sorted list of feature names, most to least important
            sorted_coeffs : the coefficients sorted by absolute value
    """
    import warnings
    warnings.filterwarnings("ignore")

    st = StandardScaler()
    x_normed = st.fit_transform(x)

    #BestLASSO = grid_search(x_normed, y, Lasso(), param_grid={"alpha": np.logspace(-14, 4, 50)}, verbose=False)
    BestLASSO = Lasso(alpha=alpha)

    BestLASSO.fit(x, y)
    coeff = BestLASSO.coef_

    num_features = len(feature_names)

    #create dictionary
    f_imps = {}
    f_imps_abs = {}
    for i in range(num_features):
        f_imps[feature_names[i]] = coeff[i]
        f_imps_abs[feature_names[i]] = np.abs(coeff[i])

    #sort dictionary by absolute value of coefficient
    sorted_feature_names = sorted(f_imps_abs, key=f_imps_abs.__getitem__, reverse=True)
    sorted_coeffs = [f_imps[name] for name in sorted_feature_names]

    if (print_latex): fi_print_latex(sorted_feature_names, sorted_coeffs, name="LASSO", num_to_print=num_to_print)

    return sorted_feature_names, sorted_coeffs


#------------------------------------------------------------------------------------------
def shuffle_importance(x, y, feature_names, model=RandomForestRegressor(n_estimators = 50),
                          num_to_print='all', print_latex=False):
    """
        Feature importance by looking at accuracy drop after shuffling
        Required arguments:
            x : data matrix (number samples x number of features), NumPy array
            y : target property vector, as NumPy array
            feature_names : list of feature names
        Optional arguments:
            model : a scikit-learn model, if none is supplied random forest is used.
        returns:
            sorted_feature_names : sorted list of feature names, most to least important
            sorted_importances : the importances  sorted by absolute value
    """

    model = model.fit(x, y)

    scores = defaultdict(list)

    num_features = x.shape[1]

    #cross validate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(10, test_size=.4):
        X_train, X_test = x[train_idx], x[test_idx]
        Y_train, Y_test = y[train_idx], y[test_idx]
        model = model.fit(X_train, Y_train)
        MAE = mean_absolute_error(Y_test, model.predict(X_test))
        for i in range(num_features):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            scores[feature_names[i]] += [mean_absolute_error(Y_test,model.predict(X_t))/MAE]

    #create dictionary
    f_imps = {}
    f_imps_abs = {}

    for i in range(num_features):
        this_avg_score = np.mean(scores[feature_names[i]])
        f_imps[feature_names[i]] = this_avg_score
        f_imps_abs[feature_names[i]] = np.abs(this_avg_score)

    #sort dictionary by absolute value of coefficient
    sorted_feature_names = sorted(f_imps_abs, key=f_imps_abs.__getitem__, reverse=True)
    sorted_values = [f_imps[name] for name in sorted_feature_names]

    if (print_latex): fi_print_latex(sorted_feature_names, sorted_values, name="shuffle feature importance analysis")

    return sorted_feature_names, sorted_values

#------------------------------------------------------------------------------------------
def stability_selection_with_LASSO(x, y, feature_names, print_latex=False, alpha=.01):
    import warnings
    warnings.filterwarnings("ignore")

    st = StandardScaler()
    x_normed = st.fit_transform(x)

    #best_lasso = grid_search(x_normed, y, Lasso(), param_grid={"alpha": np.logspace(-14, 4, 20)}, verbose=True)
    #best_alpha = best_lasso.get_params()['alpha']

    rlasso = RandomizedLasso(alpha=alpha)

    rlasso.fit(x, y)
    scores = rlasso.scores_

    num_features = len(feature_names)

    #create dictionary
    f_imps = {}
    f_imps_abs = {}

    for i in range(num_features):
        f_imps[feature_names[i]] = scores[i]

    #sort dictionary by absolute value of coefficient
    sorted_feature_names = sorted(f_imps, key=f_imps.__getitem__, reverse=True)
    sorted_values = [f_imps[name] for name in sorted_feature_names]

    if (print_latex): fi_print_latex(sorted_feature_names, sorted_values, name="stability selection")

    return sorted_feature_names, sorted_values


#------------------------------------------------------------------------------------------
def sort_scores(scores, feature_names):

    num_features = len(feature_names)

    #create dictionary
    f_imps = {}
    f_imps_abs = {}

    for i in range(num_features):
        f_imps[feature_names[i]] = scores[i]
        f_imps_abs[feature_names[i]] = np.abs(scores[i])

    #sort dictionary by absolute value of coefficient
    sorted_feature_names = sorted(f_imps, key=f_imps_abs.__getitem__, reverse=True)
    sorted_values = [f_imps[name] for name in sorted_feature_names]

    return sorted_feature_names, sorted_values


#------------------------------------------------------------------------------------------
def pearson_correlation(x, y, feature_names):

    from scipy.stats import pearsonr

    num_features = len(feature_names)

    scores = np.zeros([num_features])

    for i in range(num_features):
        r_value, p_value = pearsonr(x[:,i], y)
        if (p_value < .01):
            scores[i] = r_value

    return sort_scores(scores, feature_names)

#------------------------------------------------------------------------------------------
def f_test(x, y, feature_names, p_cutoff = 0.05):

    from sklearn.feature_selection import f_regression

    num_features = len(feature_names)

    raw_scores, p_values = f_regression(x, y)

    scores = np.zeros(num_features)

    #just keep non statistically significant scores at 0
    for i in range(num_features):
        if (p_value[i] < p_cutoff):
            scores[i] = raw_scores[i]

    return sort_scores(scores, feature_names)


#------------------------------------------------------------------------------------------
def mutual_information(x, y, feature_names):
    from sklearn.feature_selection import mutual_info_regression

    scores = mutual_info_regression(x, y)

    return sort_scores(scores, feature_names)


#------------------------------------------------------------------------------------------
def maximal_information_coefficient(x, y, feature_names):

    from minepy import MINE

    m = MINE()

    num_features = len(feature_names)

    scores = np.zeros([num_features])

    for i in range(num_features):
        m.compute_score(x[:,i], y)
        scores[i] = m.mic()

    return sort_scores(scores, feature_names)


def print_signed(coeff):
    if (coeff > 0):
        print("+%5.3f" % coeff, end='')
    elif (coeff == 0.0):
        print(" %5.3f" % coeff, end='')
    else:
        print("%5.3f" % coeff, end='')


#------------------------------------------------------------------------------------------
def compare_feature_ranking_methods(x, y, feature_names, print_latex=True,
                                    print_basic_table=False, num_to_print=10, return_dict=False):
    """
        compares a bunch of feature ranking methods, prints a latex table (optional) and then
        returns a dict with all the info (optional)
    """
    from collections import OrderedDict
    method = OrderedDict()
    method['LASSO'] = LASSO_feature_importance(x, y, feature_names, alpha=.01)
    method['LASSO stability selection'] = stability_selection_with_LASSO(x, y, feature_names, alpha=.005)
    method['random forest variance score'] = random_forest_feature_importance(x, y, feature_names)
    method['random forest shuffling'] = shuffle_importance(x, y, feature_names)
    method['Pearson correlation'] = pearson_correlation(x, y, feature_names)
    method['$f$-test'] = mutual_information(x, y, feature_names)
    method['MI'] = mutual_information(x, y, feature_names)
    method['MIC'] = maximal_information_coefficient(x, y, feature_names)

    methods = list(method.keys())
    num_methods = len(methods)


    # print basic table
    if (print_basic_table):
        for this_method in methods:
            print(this_method)
            for i in range(num_to_print):
                print(method[this_method][0][i],' ' , end ='' )
            print(method[this_method][0][10] , end ='\n' )

    #print LaTeX table
    if (print_latex):
        print("\\begin{table*}")
        print("\\begin{tabular}{c|", end='')
        for i in range(num_methods-1):
            print("cc|",end='')
        print('cc|}')

        print(" &", end='')
        for i in range(num_methods-1):
            print("\\multicolumn{2}{c|}{\\rot{"+methods[i]+"}} & ", end='')

        print("\\multicolumn{2}{c|}{\\rot{"+methods[num_methods-1]+"}}\\\\")

        for i in range(num_to_print):
            print("%i &" % (i), end='')
            for j in range(num_methods-1):
                print("%5s &" % (method[methods[j]][0][i]), end='')
                if (min(method[methods[j]][1][:])<0):
                    print_signed(method[methods[j]][1][i])
                else:
                    print("%5.3f" % method[methods[j]][1][i], end='')
                print("&", end='')

            j = num_methods-1
            print("%5s &" % (method[methods[j]][0][i]), end='')
            if (min(method[methods[j]][1][:])<0):
                print_signed(method[methods[j]][1][i])
            else:
                print("%5.3f" % method[methods[j]][1][i], end='')
            print("\\\\")

        print("\\end{tabular}")
        print("\\end{table*}")

    if (return_dict):
        return method
