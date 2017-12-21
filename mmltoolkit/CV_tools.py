from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer
from sklearn.kernel_ridge import KernelRidge
import numpy as np


#------------------- grid search ----------------------------------------
def grid_search(X, y, model, param_grid, name='', cv=KFold(n_splits=5,shuffle=True), scoring='neg_mean_absolute_error', verbose=False):
    '''Performs a grid search over param_grid and returns the best model'''
    GSmodel = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring=scoring, n_jobs=-1)

    GSmodel = GSmodel.fit(X, y)

    if (verbose):
        print(name+"best params:")
        print(GSmodel.best_params_)
        print(str(scoring)+":")
        print(-1*GSmodel.best_score_)

    return  GSmodel.best_estimator_

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))

def r2Pearson(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)
    numer = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for i in range(len(y_true)):
        numer += (y_true[i] - mean_y_true)*(y_pred[i] - mean_y_pred)
        denom1 += (y_true[i] - mean_y_true)**2
        denom2 += (y_pred[i] - mean_y_pred)**2
    return  (numer/np.sqrt(denom1*denom2+0.000000001))**2

def R2CV(y_true, y_pred):
    mean_y_true = np.mean(y)
    return 1.0 - np.mean((y_true - y_pred)**2)/np.mean((y_true-mean_y_true)**2)

def R2(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    return 1.0 - np.mean((y_true - y_pred)**2)/np.mean((y_true-mean_y_true)**2)

#--------------------------------------------------------------------
def get_scorers_dict():
    """Generate a dictionary of scoring methods that is useful for use with cross_validate()
        returns:
            scorers_dict : (dictionary)
    """

    MAPE_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    r2Pearson_scorer = make_scorer(r2Pearson, greater_is_better=True)

    scorers_dict = {'abs_err' : 'neg_mean_absolute_error',
                    'RMSE' : 'neg_mean_squared_error',
                    'R2' : 'r2',
                    'r2P' : r2Pearson_scorer,
                    'MAPE' : MAPE_scorer}

    return scorers_dict

#--------------------------------------------------------------------
def test_model_cv(model, x, y, cv=KFold(n_splits=5,shuffle=True)):
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='neg_mean_absolute_error')

    scores = -1*scores

    return scores.mean()

#--------------------------------------------------------------------
def tune_KR_and_test(X, y, cv=KFold(n_splits=5,shuffle=True), do_grid_search=True, verbose=False):
    if (do_grid_search):
        KR_grid = {"alpha": np.logspace(-16, -2, 10),
                   "gamma": np.logspace(-15, -6, 10),
                   "kernel" : ['rbf','laplacian']}

        model = grid_search(X, y, KernelRidge(), param_grid=KR_grid, verbose=verbose)
    else:
        model = KernelRidge()

    scorers_dict = get_scorers_dict()
    scores_dict = cross_validate(model, X, y, cv=cv, n_jobs=-1, scoring=scorers_dict, return_train_score=True)

    return scores_dict
