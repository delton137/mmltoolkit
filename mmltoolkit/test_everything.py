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
from sklearn.preprocessing import StandardScaler


#----------------------------------------------------------------------------
def make_CV_models(X, y):
    '''
        performs grid searchs to find best models for dataset X, y
        parameters and models used can be changed here.
    '''

    model_dict = {
            'KRR'    : grid_search(X, y, KernelRidge(), param_grid={"alpha": np.logspace(-15, 2, 60), "gamma": np.logspace(-15, -1, 60), "kernel" : ['rbf','laplacian']}),
            'SVR'   : grid_search(X, y, SVR(), param_grid={"C": np.logspace(-1, 4, 20), "epsilon": np.logspace(-2, 2, 20)}),
            'Ridge' : grid_search(X, y, Ridge(), param_grid={"alpha": np.logspace(-6, 6, 150)} ),
            #'Lasso' : grid_search(X, y, Lasso(max_iter = 20000), param_grid={"alpha": np.logspace(-2, 6, 100)} ),
            #'BR'    : grid_search(X, y, BayesianRidge(), param_grid={"alpha_1": np.logspace(-13,-5,10),"alpha_2": np.logspace(-9,-3,10), "lambda_1": np.logspace(-10,-5,10),"lambda_2": np.logspace(-11,-4,10)}) ,
            #'GBoost': grid_search(X, y, GradientBoostingRegressor(), param_grid={"n_estimators": np.linspace(5, 350, 100).astype('int')} ),
            'RF'    : grid_search(X, y, RandomForestRegressor(), param_grid={"n_estimators": np.linspace(5, 100, 50).astype('int')}, ),
            'kNN'   : grid_search(X, y, KNeighborsRegressor(), param_grid={"n_neighbors": np.linspace(2,20,18).astype('int')} ),
            'mean'  : DummyRegressor(strategy='mean'),
            }

    return model_dict


#----------------------------------------------------------------------------
def test_everything(data, featurization_dict, targets, cv=KFold(n_splits=5,shuffle=True), verbose=False, normalize=False ):
    ''' test all combinations of target variable, featurizations, and models by performing a gridsearch CV hyperparameter
        optimization for each combination and then CV scoring the best model.
        We later changed cv=KFold() to ShuffleSplit()

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

            if (normalize):
                st = StandardScaler()
                x = st.fit_transform(x)

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
                        "r" : scores_dict['test_rP'].mean()
                }

                modelresults[modelname] = relevant_scores

                if (relevant_scores["MAE"] < best_value):
                    best[target]=[featurization, modelname]
                    best_value = relevant_scores["MAE"]

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
def print_everything(results, best, targets, boldbest=True, target_short_names=target_short_names):
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
