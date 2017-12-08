import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, ShuffleSplit
from .CV_tools import grid_search, get_scorers_dict
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge


#----------------------------------------------------------------------------
def test_featurizations_and_plot(featurization_dict, y, cv=KFold(n_splits=5,shuffle=True),
                                make_plots=False, save_plot=False, verbose=False, target_prop_name='',
                                units = '', make_combined_plot=False):
    ''' test a bunch of models and print out a sorted list of CV accuracies
        inputs:
            x: training data features, numpy array or Pandas dataframe
            y: training data labels, numpy array or Pandas dataframe
            model_dict: a dictionary of the form {name : model()}, where 'name' is a string
                        and 'model()' is a sci-kit-learn model object.
    '''
    RMSE = {}
    mean_abs_err = {}
    mean_abs_err_train = {}
    std_abs_err_train = {}
    std_abs_err = {}
    mean_MAPE = {}
    mean_R2train = {}
    mean_R2test = {}
    mean_r2Ptest = {}
    mean_r2Ptrain = {}
    percent_errors = {}
    model_dict = {}
    subplot_index = 1

    num_featurizations = len(featurization_dict.keys())

    num_fig_rows = 5
    num_fig_columns = np.ceil((num_featurizations+1)/num_fig_rows)

    if (make_combined_plot | make_plots):
        plt.clf()
        plt.figure(figsize=(6*num_fig_columns,6*num_fig_rows))

    for (name, x) in featurization_dict.items():
        if (verbose): print("running %s" % name)

        if (x.ndim == 1):
            x = x.reshape(-1,1)

        #------ model selection & grid search ----
        grid = np.concatenate([np.logspace(-14, -2, 12),np.logspace(-2, 2, 200)])
        KR_grid = {"alpha": np.logspace(-16, -2, 50),
                         "gamma": np.logspace(-15, -6, 10),
                        "kernel" : ['rbf','laplacian']}
        #model = grid_search(x, y, Lasso(), cv=cv, param_grid={"alpha": grid }, verbose=True)
        model = grid_search(x, y, KernelRidge(), param_grid=KR_grid, verbose = True)
        #model = KernelRidge(**{'alpha': 9.8849590466255858e-11, 'gamma': 1.7433288221999873e-11, 'kernel': 'rbf'})
        #model = grid_search(x, y,SVR(), param_grid={"C": np.logspace(-1, 3, 40), "epsilon": np.logspace(-2, 1, 40)}, name = "SVR", verbose=True, cv=cv)
        #model = grid_search(x, y, RandomForestRegressor(), param_grid={"n_estimators": np.linspace(10, 50,5).astype('int')}, verbose=True)
        #model = BayesianRidge()

        scorers_dict = get_scorers_dict()

        scores_dict = cross_validate(model, x, y, cv=cv, n_jobs=-1, scoring=scorers_dict, return_train_score=True)
        RMSE[name] = np.sqrt(-1*scores_dict['test_RMSE'].mean())
        mean_MAPE[name] = -1*scores_dict['test_MAPE'].mean()
        mean_abs_err_train[name] = -1*scores_dict['train_abs_err'].mean()
        mean_abs_err[name] = -1*scores_dict['test_abs_err'].mean()
        std_abs_err_train[name] = np.std(-1*scores_dict['train_abs_err'])
        std_abs_err[name] = np.std(-1*scores_dict['test_abs_err'])
        mean_R2test[name] = scores_dict['test_R2'].mean()
        mean_R2train[name] = scores_dict['train_R2'].mean()
        mean_r2Ptrain[name] = scores_dict['train_r2P'].mean()
        mean_r2Ptest[name] = scores_dict['test_r2P'].mean()
        model_dict[name] = model


    sorted_names = sorted(mean_abs_err, key=mean_abs_err.__getitem__, reverse=False)

    if (make_plots):
        for name in sorted_names:
            x = featurization_dict[name]
            if (x.ndim == 1):
                x = x.reshape(-1,1)
            model = model_dict[name]
            ax = plt.subplot(num_fig_rows, num_fig_columns, subplot_index)
            subplot_index += 1
            plt.xlabel('Actual '+target_prop_name, fontsize=19)
            plt.ylabel('Predicted '+target_prop_name, fontsize=19)
            #label = '\n mean % error: '+str(mean_MAPE[name])
            label=name+'\n'+r'$\langle$MAE$\rangle$ (test) = '+" %4.2f "%(mean_abs_err[name])+units+"\n"+r'$\langle r\rangle$ (test) = %4.2f'%(mean_r2Ptest[name])
            plt.text(.05, .72, label, fontsize = 21, transform=ax.transAxes)

            kf = cv
            train, test = kf.split(x).__next__() #first in the generator
            model.fit(x[train], y[train])
            y_pred_test = model.predict(x[test])
            y_pred_train = model.predict(x[train])
            plt.scatter(y[test],y_pred_test, label = 'Test', c='blue',alpha = 0.7)
            plt.scatter(y[train],y_pred_train, label = 'Train', c='lightgreen',alpha = 0.7)
            plt.legend(loc=4, fontsize=21)

            #square axes
            maxy = 1.05*max(y)
            plt.plot([0,maxy],[0, maxy],'k-')
            #reference line
            plt.xlim([0,maxy])
            plt.ylim([0,maxy])

    plt.tight_layout()
    if (save_plot): plt.savefig('model_comparison.pdf')
    plt.show()


    print("\\begin{tabular}{c c c c c c c c c}")
    print("                   name          & MAE_{\\ff{train}}   &  MAE_{\\ff{test}}  & MAPE_{\\ff{test}} & RMSE_{\\ff{test}}  & R^2_{\\ff{train}} &  R^2_{\\ff{test}} &  r_{\\ff{train}} & r_{\\ff{test}}          \\\\  ")
    print("\\hline")
    for i in range(len(sorted_names)):
        name = sorted_names[i]
        print("%30s &   %5.3f $\\pm$ %3.2f & %5.3f $\\pm$ %3.2f & %5.2f &  %5.3f &  %5.2f & %5.2f & %5.2f & %5.2f  \\\\" % (name,
                                                        mean_abs_err_train[name],
                                                        std_abs_err_train[name],
                                                        mean_abs_err[name],
                                                        std_abs_err[name],
                                                        mean_MAPE[name],
                                                        RMSE[name],
                                                        mean_R2train[name],
                                                        mean_R2test[name],
                                                        mean_r2Ptrain[name],
                                                        mean_r2Ptest[name]))
    print("\\end{tabular}")
