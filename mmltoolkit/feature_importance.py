
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .CV_tools import grid_search
import numpy as np

#------------------------------------------------------------------------------------------
def fi_print_latex(sorted_feature_names, sorted_coeffs, name="", num_to_print='all'):
    """helper function to print a LaTeX table of feature importances in a pretty format
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
    """ Feature importance by mean decrease accuracy.
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
    rf = rf.fit(x, y)

    from sklearn.cross_validation import ShuffleSplit
    from sklearn.metrics import r2_score
    from collections import defaultdict

    X = boston["data"]
    Y = boston["target"]

    rf =
    scores = defaultdict(list)

    #crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    print "Features sorted by their score:"
    print sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True)

    #sort dictionary
    sorted_feature_names = sorted(f_imps, key=f_imps.__getitem__, reverse=True)
    sorted_values = sorted(f_imps.values(), reverse=True)

    if (print_latex): fi_print_latex(sorted_feature_names, sorted_values, name="random forest mean decrease impurity")

    return sorted_feature_names, sorted_values

#------------------------------------------------------------------------------------------
def random_forest_feature_importance(x, y, feature_names, num_to_print='all', print_latex=True):
    """ Calculation of feature importance scikit-learn random forest feature importance
        REFS:
            Gilles Louppe, et al. "Understanding variable importances in forests of randomized trees"
        AKA "gini importance" or "mean decrease (gini) impurity" or "mean decrease gini"
        Required arguments:
            x : data matrix (number samples x number of features), NumPy array
            y : target property vector, as NumPy array
            feature_names : list of feature names
        returns:
            sorted_feature_names : sorted list of feature names, most to least important
            sorted_importances  : the importances sorted by absolute value
    """
    #rf = grid_search(x, y, RandomForestRegressor(), param_grid={"n_estimators": np.linspace(10, 120, 12).astype('int')}, verbose=False)
    rf =   RandomForestRegressor( n_estimators= 100 ) #we want to ues a lot of regressors !
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
def LASSO_feature_importance(x, y, feature_names, print_latex=True, num_to_print='all'):
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
    st = StandardScaler()
    x_normed = st.fit_transform(x)

    BestLASSO = grid_search(x_normed, y, Lasso(), param_grid={"alpha": np.logspace(-15, 4, 100)}, verbose=False)

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
