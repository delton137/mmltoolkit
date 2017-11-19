from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np


#------------------- grid search ----------------------------------------
def grid_search(X, y, model, param_grid, name='', cv=KFold(n_splits=5,shuffle=True), scoring='neg_mean_absolute_error', verbose=False):
    '''Performs a grid search over param_grid and returns the best model'''
    GSmodel = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring=scoring, n_jobs=-1)

    GSmodel = GSmodel.fit(X, y)

    if (verbose):
        print(name+"best params:")
        print(GSmodel.best_params_)
        #print(scoring+":")
        #print(-1*GSmodel.best_score_)

    return  GSmodel.best_estimator_

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))

def R2CV(y_true, y_pred):
    mean_y_true = np.mean(y)
    #mean_y_true = np.mean(y_true)
    return 1.0 - np.mean((y_true - y_pred)**2)/np.mean((y_true-mean_y_true)**2)

def r2Pearson(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)
    numer = 0
    denom1 = 0
    denom2 = 0
    for i in range(len(y_true)):
        numer += (y_true[i] - mean_y_true)*(y_pred[i] - mean_y_pred)
        denom1 += (y_true[i] - mean_y_true)**2
        denom2 += (y_pred[i] - mean_y_pred)**2
    return  (numer/np.sqrt(denom1*denom2))**2

def get_scorers_dict():
    """Generate a dictionary of scoring methods that is useful for use with cross_validate()
        returns:
            scorers_dict : (dictionary)
    """

    MAPE_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    R2CV_scorer = make_scorer(R2CV, greater_is_better=True)

    r2Pearson_scorer = make_scorer(r2Pearson, greater_is_better=True)

    scorers_dict = {'abs_err' : 'neg_mean_absolute_error',
                    'RMSE' : 'mean_squared_error',
                    'R2' : 'r2',
                    'r2P' : r2Pearson_scorer,
                    'R2CV': R2CV_scorer,
                    'MAPE' : MAPE_scorer}

    return scorers_dict
