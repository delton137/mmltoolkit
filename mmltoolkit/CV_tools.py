from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.kernel_ridge import KernelRidge
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------
def grid_search(X, y, model, param_grid, name='', cv=KFold(n_splits=5,shuffle=True),
                scoring='neg_mean_absolute_error', verbose=False, n_jobs=-1):
    '''Performs a grid search over param_grid and returns the best model'''

    GSmodel = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring=scoring, n_jobs=n_jobs)

    GSmodel = GSmodel.fit(X, y)

    if (verbose):
        print(name+"best params:")
        print(GSmodel.best_params_)
        print(str(scoring)+":")
        print(-1*GSmodel.best_score_)

    return GSmodel.best_estimator_

#------------------------------------------------------------------------
def nested_grid_search_CV(X, y, model, param_grid, name='', inner_cv=KFold(n_splits=5, shuffle=True),
                         outer_cv=ShuffleSplit(), verbose=False, n_jobs=-1):
    '''
        This is the standard "nested" CV scheme, where hyperparameter optimization is redone
        for each train_test split by splitting the train set further into train and validation sets.
    '''

    kf = outer_cv
    n_folds = outer_cv.get_n_splits()

    scores_dict = {'MAE' : 0, 'MAE_std' : 0, 'RMSE' : 0,
                    'R2' : 0, 'rP' : 0, 'MAPE' : 0
                    'rP_train' : 0 , 'MAE_train' : 0 }

    MAEs = []

    for i in range(n_folds):

        train, test = kf.split(X).__next__()

        best_model = grid_search(X[train], y[train], model, param_grid, cv=inner_cv), n_jobs=-1)

        best_model.fit(X[train], y[train])

        y_pred =  best_model.predict(X[test])
        y_true = y[test]

        y_pred_train =  best_model.predict(X[train])
        y_true_train = y[train]

        MAEs += [mean_absolute_error(y_true, y_pred)]
        scores_dict['MAE_train'] += mean_absolute_error(y_true_train, y_pred_train)
        scores_dict['RMSE'] += root_mean_squared_error(y_true, y_pred)
        scores_dict['R2']   += R2(y_true, y_pred)
        scores_dict['rP']   += rPearson(y_true, y_pred)
        scores_dict['rP_train'] += rPearson(y_true_train, y_pred_train)
        scores_dict['MAPE'] += mean_absolute_percentage_error(y_true, y_pred)

    scores_dict['MAE']   = np.mean(MAEs)
    scores_dict['MAE_std'] = np.std(MAEs)
    scores_dict['MAE_train'] = scores_dict['MAE_train']/n_folds
    scores_dict['RMSE'] = scores_dict['RMSE']/n_folds
    scores_dict['R2']   = scores_dict['R2']/n_folds
    scores_dict['rP']   = scores_dict['rP']/n_folds
    scores_dict['rP_train'] = scores_dict['rP_train']/n_folds
    scores_dict['MAPE'] = scores_dict['MAPE']/n_folds

    return scores_dict

#------------------------------------------------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))

def neg_mean_absolute_error(y_true, y_pred):
    return -1*np.mean(np.abs((y_true - y_pred)))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def neg_mean_squared_error(y_true, y_pred):
    return -1*np.mean((y_true - y_pred)**2)

def rPearson(y_true, y_pred):
    """ calculates Pearson correlation r (not r^2) """
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

def r2(y_true, y_pred):
    """ another function for Pearson r included for backwards compatibility"""
    return rPearson(y_true, y_pred)

def R2(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    return 1.0 - np.mean((y_true - y_pred)**2)/np.mean((y_true-mean_y_true)**2)

#--------------------------------------------------------------------
def get_scorers_dict():
    """ Generate a dictionary of scikitlearn scoring methods that is useful for use with cross_validate()
        returns:
            scorers_dict : (dictionary)
    """

    MAPE_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    rPearson_scorer = make_scorer(rPearson, greater_is_better=True)

    scorers_dict = {'abs_err' : 'neg_mean_absolute_error',
                    'RMSE' : 'neg_mean_squared_error',
                    'R2' : 'r2',
                    'rP' : rPearson_scorer,
                    'MAPE' : MAPE_scorer}

    return scorers_dict

#--------------------------------------------------------------------
def get_score_functions_dict():
        """Generate a dictionary of scoring methods that is useful for use with cross_validate()
            returns:
                scorers_dict : (dictionary)
        """

        scorers_dict = {'abs_err' : mean_absolute_error,
                        'RMSE' : root_mean_squared_error,
                        'R2' : R2,
                        'rP' : rPearson,
                        'MAPE' : mean_absolute_percentage_error
                        }

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


#--------------------------------------------------------------------
def test_and_plot(X, y, model, splitter=ShuffleSplit(n_splits=5), groups=None, units='', property_name='', display_plot = True, plot_title='', savefig=False, filename=''):



    scorers_dict = get_score_functions_dict()

    n_splits = splitter.get_n_splits()

    scores = defaultdict(list)

    for n in range(n_splits):
        for train_index, test_index in splitter.split(X, groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred = model.predict(X_test)

            for scorer in list(scorers_dict.keys()):
                scores['test_'+scorer] += [scorers_dict[scorer](y_test,y_pred)]
                scores['train_'+scorer] += [scorers_dict[scorer](y_train,y_pred_train)]


    MAE_test = np.mean(scores['test_abs_err'])
    r_test = np.mean(scores['test_rP'])

    if (display_plot):
        plt.figure(figsize=(7,7))
        plt.clf()
        plt.xlabel('Actual '+property_name,fontsize=19)
        plt.ylabel('Predicted '+property_name, fontsize=19)
        MAE_test = 10**MAE_test
        if (units == ''):
            label=plot_title+'\n'+r'$\langle$MAE$\rangle$ (test) = '+" %4.2f "%(MAE_test)+"\n"+r'$\langle r\rangle$ (test) = %4.2f'%(r_test)
        else:
            label=plot_title+'\n'+r'$\langle$MAE$\rangle$ (test) = '+" %4.2f "%(MAE_test)+'('+units+')'+"\n"+r'$\langle$ r$\rangle$ (test) = %4.2f'%(r_test)
        ax = plt.gca()
        plt.text(.05, .85, label, fontsize = 21, transform=ax.transAxes)

        train, test = splitter.split(X, groups).__next__() #first in the generator
        model.fit(X[train], y[train])
        y_pred_test = model.predict(X[test])
        y_pred_train = model.predict(X[train])
        plt.scatter(y[test],y_pred_test, label = 'Test', c='blue',alpha = 0.7)
        plt.scatter(y[train],y_pred_train, label = 'Train', c='lightgreen',alpha = 0.7)
        plt.legend(loc=4, fontsize=21)

        #square axes
        maxy = 1.05*max(y)
        #square axes
        maxy = 1.05*max([max(y_pred_train), max(y_pred_test), max(y)])
        miny = .95*min([min(y_pred_train), min(y_pred_test), min(y)])

        #reference line
        plt.plot([miny,maxy],[miny, maxy],'k-')

        if (savefig):
            plt.savefig(filename+".png",bbox_inches='tight', pad_inches=0)

        plt.show()


    return scores
