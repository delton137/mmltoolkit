import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from .CV_tools import get_scorers_dict, grid_search
from sklearn.dummy import DummyRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate



#----------------------------------------------------------------------------
def make_CV_models(X, y):
    '''
        performs grid searchs to find best models for dataset X, y
        parameters and models used can be changed here.
    '''
    Best_Ridge = grid_search(X, y,Ridge(),
                  param_grid={"alpha": np.logspace(-2, 6, 100)}, name = "Ridge")

    #Best_Lasso = grid_search(X, y,Lasso(max_iter = 20000),
    #              param_grid={"alpha": np.logspace(-2, 6, 100)}, name = "Lasso")

    Best_RandomForestRegressor = grid_search(X, y,RandomForestRegressor(),
                  param_grid={"n_estimators": np.linspace(5, 200, 100).astype('int')}, name = "Random Forrest")

    #Best_GradientBoostingRegressor = grid_search(X, y,GradientBoostingRegressor(),
    #              param_grid={"n_estimators": np.linspace(5, 350, 100).astype('int')}, name = "Gradient Boosting")

    Best_SVR = grid_search(X, y,SVR(),
                  param_grid={"C": np.logspace(-1, 4, 20),
                             "epsilon": np.logspace(-2, 2, 20)}, name = "SVR ")

    Best_KNeighborsRegressor = grid_search(X, y,KNeighborsRegressor(),
                  param_grid={"n_neighbors": np.linspace(2,20,18).astype('int')}, name= "KNN")

    #Best_BayesianRidge = grid_search(X, y,BayesianRidge(),
    #             param_grid={"alpha_1": np.logspace(-13,-5,10),"alpha_2": np.logspace(-9,-3,10),
    #                       "lambda_1": np.logspace(-10,-5,10),"lambda_2": np.logspace(-11,-4,10)}, name= "Bayesian Ridge")

    Best_KernelRidge = grid_search(X, y,KernelRidge(),
                  param_grid={"alpha": np.logspace(-15, 1, 50),
                 "gamma": np.logspace(-15, -5, 50), "kernel" : ['rbf','laplacian']}, name = "Kernel Ridge")

    model_dict = {
            'KR': Best_KernelRidge,
            'SVR': Best_SVR,
            'Ridge':Best_Ridge,
            #'Lasso':Best_Lasso,
            #'BR': Best_BayesianRidge,
            #'GBoost': Best_GradientBoostingRegressor,
            'RF': Best_RandomForestRegressor,
            'kNN': Best_KNeighborsRegressor,
            'mean': DummyRegressor(strategy='mean'),
            }

    return model_dict


#----------------------------------------------------------------------------
def test_everything(data, featurization_dict, targets, cv=KFold(n_splits=5,shuffle=True), verbose=False ):
    ''' test all combinations of target variable, featurizations, and models by performing a gridsearch CV hyperparameter
        optimization for each combination and then CV scoring the best model.

        required args:
            data : a pandas dataframe with the target data in columns
            featurization_dict : a dictionary of the form {"featurization name" : X_featurization }
            targets : a list of target names, corresponding to the columns in data
        returns:
            results : a nested dictionary of the form
                        {target: { featurization_name: {model_name: scores_dict{ 'MAE': value, 'r2':value, etc }}}}
            best : a dictionary of the form {target : [best_featurization_name, best_model_name]}
    '''

    results={}  #nested dict
    best={}

    num_targets = len(targets)
    scorers_dict = get_scorers_dict()

    for target in targets:
        if (verbose): print("running target %s" % target)

        y = np.array(data[target].values)

        featurizationresults = {}

        best_value = 1000000000000

        for featurization in featurization_dict.keys():
            if (verbose): print("    doing hyperparameter search for featurization %s" % featurization)

            x = featurization_dict[featurization]

            if (x.ndim == 1):
                x = x.reshape(-1,1)

            model_dict = make_CV_models(x, y)

            modelresults = {}

            for modelname in model_dict.keys():

                model = model_dict[modelname]
                scores_dict = cross_validate(model, x, y, cv=cv, n_jobs=-1,
                                             scoring=scorers_dict, return_train_score=True)

                relevant_scores={
                        "train_MAE": -1*scores_dict['train_abs_err'].mean(),
                        "MAPE": -1*scores_dict['test_MAPE'].mean(),
                        "MAE": -1*scores_dict['test_abs_err'].mean(),
                        "MAE_std": np.std(-1*scores_dict['test_abs_err']),
                        "r2" : scores_dict['test_r2P'].mean()
                }

                modelresults[modelname] = relevant_scores

                if (relevant_scores["MAE"] < best_value):
                    best[target]=[featurization, modelname]
                    best_value = relevant_scores["MAE"]

            featurizationresults[featurization] = modelresults

        results[target] = featurizationresults

    return (results, best)
