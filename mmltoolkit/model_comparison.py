import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from .CV_tools import *



#------------------- mdoel comparison function ---------------------------------
def test_models_and_plot(x, y, model_dict, cv=KFold(n_splits=5,shuffle=True), make_plots=False, save_plot=False,
                         target_prop_name='', units = '', verbose=False, make_combined_plot=False, num_fig_rows = 2):
    ''' test a bunch of models (with hyperparameters already set) and print out a sorted list of CV metrics
        inputs:
            x: training data features, numpy array or Pandas dataframe
            y: training data labels, numpy array or Pandas dataframe
            model_dict: a dictionary of the form {name : model()}, where 'name' is a string
                        and 'model()' is a sci-kit-learn model object.
    '''
    scorers_dict = get_scorers_dict()

    RMSE = {}
    mean_abs_err = {}
    mean_abs_err_train = {}
    std_abs_err_train = {}
    std_abs_err = {}
    mean_MAPE = {}
    mean_R2train = {}
    mean_R2test = {}
    mean_rPtest = {}
    mean_rPtrain = {}
    percent_errors = {}
    subplot_index = 1

    num_models = len(model_dict.keys())
    num_fig_columns = np.ceil((num_models)/num_fig_rows)

    if (make_combined_plot | make_plots):
        plt.figure(figsize=(6*num_fig_columns,6*num_fig_rows))
        plt.clf()

    for (name, model) in model_dict.items():
        if (verbose): print("running %s" % name)

        scores_dict = cross_validate(model, x, y, cv=cv, n_jobs=-1, scoring=scorers_dict, return_train_score=True)
        RMSE[name] = np.sqrt(-1*scores_dict['test_RMSE'].mean())
        mean_MAPE[name] = -1*scores_dict['test_MAPE'].mean()
        mean_abs_err_train[name] = -1*scores_dict['train_abs_err'].mean()
        mean_abs_err[name] = -1*scores_dict['test_abs_err'].mean()
        std_abs_err_train[name] = np.std(-1*scores_dict['train_abs_err'])
        std_abs_err[name] = np.std(-1*scores_dict['test_abs_err'])
        mean_R2test[name] = scores_dict['test_R2'].mean()
        mean_R2train[name] = scores_dict['train_R2'].mean()
        mean_rPtrain[name] = scores_dict['train_rP'].mean()
        mean_rPtest[name] = scores_dict['test_rP'].mean()
        model_dict[name] = model

    sorted_names = sorted(mean_abs_err, key=mean_abs_err.__getitem__, reverse=False)


    if (make_plots):
        for name in sorted_names:
            model = model_dict[name]
            ax = plt.subplot(num_fig_rows, num_fig_columns, subplot_index)
            subplot_index += 1
            plt.xlabel('Actual '+target_prop_name, fontsize=19)
            plt.ylabel('Predicted '+target_prop_name, fontsize=19)
            #label = '\n mean % error: '+str(mean_MAPE[name])
            label=name+'\n'+r'$\langle$MAE$\rangle$ (test) = '+" %4.2f "%(mean_abs_err[name])+units+"\n"+r'$\langle r\rangle$ (test) = %4.2f'%(mean_rPtest[name])
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
            #square axes
            maxy = 1.05*max([max(y_pred_train), max(y_pred_test), max(y)])
            miny = .95*min([min(y_pred_train), min(y_pred_test), min(y)])

            #reference line
            plt.plot([miny,maxy],[miny, maxy],'k-')
            #plt.xlim([miny,maxy])
            #plt.ylim([miny,maxy])

    plt.tight_layout()
    if (save_plot): plt.savefig('model_comparison'+title.strip()+'.pdf')
    plt.show()

    print("\\begin{tabular}{c c c c c c c c c}")
    print("                   name          & MAE$_{\\ff{train}}$   &  MAE$_{\\ff{test}}$  & MAPE$_{\\ff{test}}$ & RMSE$_{\\ff{test}}$  & $R^2_{\\ff{train}}$ &  $R^2 _{\\ff{test}} $&  $r_{\\ff{train}}$ & $r_{\\ff{test}} $         \\\\  ")
    print("\\hline")
    for i in range(len(sorted_names)):
        name = sorted_names[i]
        print("%30s &   %5.2f $\\pm$ %3.2f & %5.2f $\\pm$ %3.2f & %5.2f &  %5.3f &  %5.2f & %5.2f & %5.2f & %5.2f  \\\\" % (name,
                                                        mean_abs_err_train[name],
                                                        std_abs_err_train[name],
                                                        mean_abs_err[name],
                                                        std_abs_err[name],
                                                        mean_MAPE[name],
                                                        RMSE[name],
                                                        mean_R2train[name],
                                                        mean_R2test[name],
                                                        mean_rPtrain[name],
                                                        mean_rPtest[name]))
    print("\\end{tabular}")
