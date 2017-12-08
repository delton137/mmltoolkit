from sklearn.model_selection import KFold, ShuffleSplit

#----------------------------------------------------------------------------
def test_everything(data, featurization_dict, targets, cv=KFold(n_splits=5,shuffle=True), verbose=False ):
    ''' test all combinations of target variable, featurization, and model by performing a gridsearch CV hyperparameter
        optimization for each combination and then a CV scoring for the best model.

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
