import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from .CV_tools import get_scorers_dict, grid_search, nested_grid_search_CV
from sklearn.dummy import DummyRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

#----------------------------------------------------------------------------
def make_models():
    ''' return dictionary with models and their corresponding parameter grids for hyperparameter optimizaiton '''

    model_dict = {
            'KRR'   :{ 'model' : KernelRidge(), 'param_grid' : {"alpha": np.logspace(-14, 2, 200), "gamma": np.logspace(-14, -2, 100), "kernel" : ['rbf']}},
            #'SVR'   :{ 'model' : SVR(), 'param_grid' : {"C": np.logspace(-1, 4, 20), "epsilon": np.logspace(-2, 2, 20)}},
            #'Ridge' :{ 'model' : Ridge(), 'param_grid' : {"alpha": np.logspace(-6, 6, 150)}},
            #'Lasso' :{ 'model' : Lasso(max_iter = 20000), 'param_grid' : {"alpha": np.logspace(-2, 6, 100)}},
            #'BR'    :{ 'model' : BayesianRidge(), 'param_grid' : {"alpha_1": np.logspace(-13,-5,10),"alpha_2": np.logspace(-9,-3,10), "lambda_1": np.logspace(-10,-5,10),"lambda_2": np.logspace(-11,-4,10)}},
            #'GBoost':{ 'model' : GradientBoostingRegressor(), 'param_grid' : {"n_estimators": np.linspace(5, 350, 100).astype('int')}},
            #'RF'     :{ 'model' : RandomForestRegressor(), 'param_grid' : {"n_estimators": np.linspace(5, 100, 50).astype('int')}},
            #'kNN'   :{ 'model' : KNeighborsRegressor(), 'param_grid' : {"n_neighbors": np.linspace(2,20,18).astype('int')}},
            #'mean'  :{ 'model' : DummyRegressor(strategy='mean'), 'param_grid' : {} },
            }

    return model_dict

#----------------------------------------------------------------------------
def make_CV_models(X, y):
    '''performs grid searches to find all the best models for dataset X, y'''

    model_dict = {
            'KRR'    : grid_search(X, y, KernelRidge(), param_grid={"alpha": np.logspace(-10, 2, 300), "gamma": np.logspace(-10, -1, 100), "kernel" : ['rbf']}),
            'SVR'   : grid_search(X, y, SVR(), param_grid={"C": np.logspace(-1, 4, 20), "epsilon": np.logspace(-2, 2, 20)}),
            'Ridge' : grid_search(X, y, Ridge(), param_grid={"alpha": np.logspace(-6, 6, 150)} ),
            'Lasso' : grid_search(X, y, Lasso(max_iter = 20000), param_grid={"alpha": np.logspace(-2, 6, 100)} ),
            'BR'    : grid_search(X, y, BayesianRidge(), param_grid={"alpha_1": np.logspace(-13,-5,10),"alpha_2": np.logspace(-9,-3,10), "lambda_1": np.logspace(-10,-5,10),"lambda_2": np.logspace(-11,-4,10)}) ,
            'GBoost': grid_search(X, y, GradientBoostingRegressor(), param_grid={"n_estimators": np.linspace(5, 350, 100).astype('int')} ),
            'RF'    : grid_search(X, y, RandomForestRegressor(), param_grid={"n_estimators": np.linspace(5, 100, 50).astype('int')}, ),
            'kNN'   : grid_search(X, y, KNeighborsRegressor(), param_grid={"n_neighbors": np.linspace(2,20,18).astype('int')} ),
            'mean'  : DummyRegressor(strategy='mean'),
            }

    return model_dict


#----------------------------------------------------------------------------
def test_everything(data, featurization_dict, targets, inner_cv=KFold(n_splits=5,shuffle=True),
                    outer_cv=ShuffleSplit(n_splits=20, test_size=0.2), verbose=False, normalize=False ):
    '''
        test all combinations of target variable, featurizations, and models
        by performing a gridsearch CV hyperparameter
        optimization for each combination and then CV scoring the best model.

        required args:
            data : a pandas dataframe with data for the different targets in the columns
            featurization_dict : a dictionary of the form {"featurization name" : X_featurization }, where X_featurization is the data array
            targets : a list of target names, corresponding to the columns in data
        important optional args:
            outer_cv : crossvalidation object specifying cross validation strategy for the outer
                     train-test CV loop. Typically we choose ShuffleSplit with a large # of splits.
            inner_cv : crossvalidation object specifying cross validation strategy for the inner train-validation
                     CV loop. K-fold with 5 folds is the standard.
        returns:
            results : a nested dictionary of the form
                     {target: { featurization_name: {model_name: scores_dict{ 'MAE': value, 'r2':value, etc }}}}
            best : a dictionary of the form {target : [best_featurization_name, best_model_name]}
    '''

    results={}
    best={}

    num_targets = len(targets)
    scorers_dict = get_scorers_dict()

    for target in targets:
        if (verbose): print("running target %s" % target)

        y = np.array(data[target].values)

        featurizationresults = {}

        best_value = 1000000000000

        for featurization in featurization_dict.keys():
            if (verbose): print("    testing featurization %s" % featurization)

            x = featurization_dict[featurization]

            x = np.array(x)

            if (x.ndim == 1):
                x = x.reshape(-1,1)

            if (normalize):
                st = StandardScaler()
                x = st.fit_transform(x)

            model_dict = make_models()

            modelresults = {}

            for modelname in model_dict.keys():

                model = model_dict[modelname]['model']
                param_grid = model_dict[modelname]['param_grid']

                scores_dict = nested_grid_search_CV(x, y, model, param_grid,
                                                    inner_cv=KFold(n_splits=5, shuffle=True),
                                                    outer_cv=outer_cv, verbose=verbose)

                modelresults[modelname] = scores_dict

                if (scores_dict['MAE'] < best_value):
                    best[target]=[featurization, modelname]
                    best_value = scores_dict["MAE"]

            featurizationresults[featurization] = modelresults

        results[target] = featurizationresults

    return (results, best)


target_short_names = {
 'Density (g/cm3)':'\\footnotesize{$\\rho ,\\frac{\\hbox{g}}{\\hbox{cc}}$ }',
 'Delta Hf solid (kj/mol)': '\\footnotesize{$\Delta H_f^{\\ff{s}} ,\\frac{\\hbox{kJ}}{\\hbox{mol}}$ }',
 'Explosive energy (kj/cc)': '\\footnotesize{$E_{\\ff{e}} ,\\frac{\\hbox{kJ}}{\\hbox{cc}}$ }',
 'Shock velocity (km/s)': '\\footnotesize{$V_{\\ff{s}} ,\\frac{\\hbox{km}}{\\hbox{s}}$ }',
 'Particle velocity (km/s)': '\\footnotesize{$V_{\\ff{p}},\\frac{\\hbox{km}}{\\hbox{s}}$ }',
 'Speed of sound (km/s)': '\\footnotesize{$V_{\\ff{snd}},\\frac{\\hbox{km}}{\\hbox{s}}$ }',
 'Pressure (Gpa)': '\\footnotesize{$P$, GPa}',
 'T(K)': '\\footnotesize{$T$, K}',
 'TNT Equiv (per cc)': '\\footnotesize{$\\frac{\\hbox{TNT}_{\\ff{equiv}}}{\\hbox{cc}}$ }'
}


#----------------------------------------------------------------------------
def print_everything(results, best, targets, boldbest=True, target_short_names=target_short_names, show_train_scores=False):

    print("\\begin{table*}[ht]")
    print("\\begin{tabular}{cc",end='')
    for l in range(len(targets)):
          print("c",end='')
    print("}")
    print(" & ",end='')
    for target in targets:
        print(" & "+target_short_names[target], end='')
    print(" \\\\")
    print("\\hline")
    featurizations = list(results[targets[0]].keys())
    models = list(results[targets[0]][featurizations[0]].keys())
    for model in models:
        for (i, featurization) in enumerate(featurizations):
            if(i == 0):
                print(model+" & ", end='')
            else:
                print(" & ", end='')
            print(featurization+" & ", end='')
            for (j, target) in enumerate(targets):
                scores_dict = results[target][featurization][model]
                #print(" %5.2f " % (scores_dict['MAE']), end='')
                #print("%4.2f" % (scores_dict['r2']), end='')
                #print(" %5.2f, %4.2f  " % (scores_dict['MAE'], scores_dict['r2']), end='')

                if (show_train_scores):
                    print("%5.2f,%5.2f" % (scores_dict['MAE'], scores_dict['train_MAE']), end='')
                else:
                    if (boldbest):
                        if ([featurization, model] == best[target]):
                            print("\\bf{%5.2f}" % (scores_dict['MAE']), end='')
                        else:
                            print("%5.2f" % (scores_dict['MAE']), end='')
                    else:
                        print("%5.2f" % (scores_dict['MAE']), end='')

                if (j == len(targets)-1):
                    print("\\\\")
                else:
                    print(" & ", end='')

    print("\\end{tabular}")
    print("\\end{table*}")
