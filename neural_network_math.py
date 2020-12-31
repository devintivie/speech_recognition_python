
import numpy as np

def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    x = np.clip(x, -100.0, 100.0)
    sig = 1 / (1 + np.exp(-x))
    return sig

def sigmoid_prime(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def tanh_prime(x):
    return 1.0 - np.tanh(x)**2

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)
        
def cost_derivative(y_preds, y_trues):
    return (y_preds - y_trues)

def calculate_zscore(input):
    '''Min max normalization
    ynew = (yold - min1)/(max1 - min1)*(max2 - min2) + min2
    '''
    [input_rows, input_cols] = input.shape
    means = np.mean(input, 0)
    vars = np.var(input, 0)
    return (means, vars)

def calculate_zscore_sent(input):
    '''Min max normalization
    ynew = (yold - min1)/(max1 - min1)*(max2 - min2) + min2
    '''
    # [input_rows, input_cols] = input.shape
    # zscore_means = np.zeros((len(input), input[0].shape[1]))
    # zscore_vars = np.zeros_like(zscore_means)
    # zscore_counts = []

    zscore_means = []
    zscore_vars = []
    zscore_counts = []

    i = 0
    for s in input:
        zscore_means.append(np.mean(s, axis=0))
        zscore_vars.append(np.var(s, axis=0))
        zscore_counts.append(len(s))
        i += 1

    mean_matrix = np.zeros((len(zscore_means), zscore_means[0].shape[0]))
    var_matrix = np.zeros((len(zscore_vars), zscore_means[0].shape[0]))
    deviation_matrix = np.zeros((len(zscore_vars), zscore_means[0].shape[0]))

    i = 0
    for n, m in zip(zscore_counts, zscore_means):
        mean_matrix[i,:] = n * m
        i += 1

    i = 0
    for n, m in zip(zscore_counts, zscore_vars):
        var_matrix[i,:] = n * m

        i += 1

    total = np.sum(zscore_counts)
    means = np.sum(mean_matrix, axis=0)
    means = means / total

    i = 0
    for n, m in zip(zscore_counts, zscore_means):
        deviation_matrix[i,:] = n * np.power(m - means, 2)

        i += 1
    deviations = np.sum(deviation_matrix, axis=0)
    vars = np.sum(var_matrix, axis=0)
    vars = (vars + deviations)/ total
    # means = [n * mean for n,mean in zip(zscore_counts, zscore_means)]

    # means = np.mean(zscore_means, axis=0)
    # vars = 
    # means = np.mean(input, 0)
    # vars = np.var(input, 0)
    return (means, vars)

def preprocess_zscore(x, means, vars):
    std =  np.sqrt(vars)

    # print(f"type(input x) = {type(x)}")

    if isinstance(x,list):
        ppdata = []
        for data in x:
            tmp = data - means
            ppdata.append(np.divide(tmp, std, out=tmp, where=std!=0))

        return ppdata
    
    elif isinstance(x, np.ndarray):
        tmp = x - means
        ppdata = np.divide(tmp, std, out=tmp, where=std!=0)#  tmp/std

        return ppdata


def preprocess_zscore_sentences(x, means, vars):
    std = np.sqrt(vars)

    ppdata = []
    for data in x:
        tmp = data - means
        ppdata.append(np.divide(tmp, std, out=tmp, where=std!=0))

    return ppdata


def calculate_min_max_norm(input, new_min, new_max):
    '''Min max normalization
    ynew = (yold - min1)/(max1 - min1)*(max2 - min2) + min2
    '''
    try:
        [input_rows, input_cols] = input.shape
    except ValueError:
        input_cols = 1
    maxes = np.max(input, 0)
    # print(self.input_maxes)
    mins = np.min(input, 0)

    return (mins, maxes)

def normalize_weights_minmax(weight_matrix_list):
    normalized = []
    new_max = 1.5
    new_min = -1.5
    for wm in weight_matrix_list:
        fix_min = False
        fix_max = False
        wmin = np.min(wm)
        wmax = np.max(wm)
        if wmin < (new_min - 0.5) :
            fix_min = True
        if wmax > (new_max + 0.5):
            fix_max = True

        if fix_max or fix_min :
            norm = (wm - wmin)/(wmax - wmin)*(new_max - new_min) + new_min
            normalized.append(norm)
            # print(f"need to fix min or max max = {wmax}, min = {wmin}")
            # print(f"new min and max, max = {np.max(norm)}, min = {np.min(norm)}, mean = {np.mean(norm)}\n")
        else:
            normalized.append(wm)   

    return normalized 

 

def preprocess_min_max(x, mins, maxes):
    pass
       # print(self.input_mins)

    # new_inputs = np.zeros_like(x, dtype=np.float)

    # for i in range(input_cols):
    #     new_inputs[:,i] = (input[:,i] - mins[i])/(maxes[i] - mins[i])*(new_max - new_min) +new_min
    #     # print(new_inputs[:,i])
    # return new_inputs   


def vectorized_result(j, vector_length):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((vector_length))
    e[j] = 1.0
    return e