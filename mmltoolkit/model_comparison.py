import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import make_scorer


#------------------- scoring functions ----------------------------------------
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

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


#------------------- mdoel comparison function ---------------------------------
def test_models_and_plot(x, y, model_dict, cv=KFold(n_splits=5,shuffle=True), make_plots=False, save_plot=False, verbose=False, make_combined_plot=False):
    ''' test a bunch of models and print out a sorted list of CV accuracies
        inputs:
            x: training data features, numpy array or Pandas dataframe
            y: training data labels, numpy array or Pandas dataframe
            model_dict: a dictionary of the form {name : model()}, where 'name' is a string
                        and 'model()' is a sci-kit-learn model object.
    '''

    MAPE_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    r2Pearson_scorer = make_scorer(r2Pearson, greater_is_better=True)


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
    subplot_index = 1

    num_models = len(model_dict.keys())
    num_fig_rows = 4
    num_fig_columns = np.ceil((num_models+1)/num_fig_rows)

    if (make_combined_plot | make_plots):
        plt.clf()
        plt.figure(figsize=(6*num_fig_columns,6*num_fig_rows))



    for (name, model) in model_dict.items():
        if (verbose): print("running %s" % name)

        scorers_dict = {'abs_err' : 'neg_mean_absolute_error',
                        'RMSE' : 'mean_squared_error',
                        'R2' : 'r2',
                        'r2P' : r2Pearson_scorer,
                        'MAPE' : MAPE_scorer}

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


        if (make_plots):
            ax = plt.subplot(num_fig_rows,num_fig_columns, subplot_index)
            subplot_index += 1
            #plt.title(name,fontsize=20)
            plt.xlabel('Actual E.E. (kJ/cc)', fontsize=19)
            plt.ylabel('Predicted E.E. (kJ/cc)', fontsize=19)
            #label = '\n mean % error: '+str(mean_MAPE[name])
            label=name+'\n'+r'$\langle$MAE$\rangle$ (test) = '+" %4.2f "%(mean_abs_err[name])+"kJ/cc\n"+r'$\langle r\rangle$ (test) = %4.2f'%(mean_r2Ptest[name])
            plt.text(.05, .72, label, fontsize = 21, transform=ax.transAxes)

            kf = cv

            if (make_combined_plot):
                for k, (train, test) in enumerate(kf.split(x,y)):
                    model.fit(x[train], y[train])
                    y_pred_test = model.predict(x[test])
                    y_pred_train = model.predict(x[train])
                    plt.scatter(y[test],y_pred_test, label = 'Test', c='blue',alpha = 0.7)
                    plt.legend(loc=4)
            else:
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
            plt.xlim([0,maxy])
            plt.ylim([0,maxy])
            #reference line


    plt.tight_layout()
    if (save_plot): plt.savefig('model_comparison.pdf')
    plt.show()

    sorted_names = sorted(mean_abs_err, key=mean_abs_err.__getitem__, reverse=False)

    print("\\begin{tabular}{c c c c c c c c c}")
    print("                   name          & MAE_{\\ff{train}}   &  MAE_{\\ff{test}}  & MAPE_{\\ff{test}} & RMSE_{\\ff{test}}  & R^2_{\\ff{train}} &  R^2_{\\ff{test}} &  r_{\\ff{train}} & r_{\\ff{test}}          \\\\  ")
    print("\\hline")
    for i in range(len(sorted_names)):
        name = sorted_names[i]
        print("%30s &   %5.2f \\pm %3.2f & %5.2f \\pm %3.2f & %5.2f &  %5.3f &  %5.2f & %5.2f & %5.2f & %5.2f  \\\\" % (name,
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
