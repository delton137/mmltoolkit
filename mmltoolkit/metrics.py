


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
