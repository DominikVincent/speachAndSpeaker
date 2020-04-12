import numpy as np
from lab2_tools import *



def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    pi = np.zeros(hmm1['startprob'].shape[0] + hmm2['startprob'].shape[0] -1)
    pi[0:hmm1['startprob'].shape[0]] = hmm1['startprob']
    tmp = np.multiply(hmm2['startprob'], hmm1['startprob'][-1])
    pi[hmm1['startprob'].shape[0]-1:] = tmp

    h1len = hmm1['startprob'].shape[0]
    h2len = hmm2['startprob'].shape[0]
    

    dim = h1len+h2len-1
    trans = np.zeros((dim, dim))
    #first transmat
    trans[:h1len-1, : h1len-1] = hmm1['transmat'][:-1, :-1]

    #transition to second mat
    #print( hmm1['transmat'])
    column = np.matrix(hmm1['transmat'][:-1,-1]).T
    #print(column)
    tile = np.tile(column, (1, h2len) )
    #print(tile)
    secondPart = np.multiply(tile, np.matrix(hmm2['startprob'][:]) )
    #print(secondPart)
    secondPart = secondPart
    trans[:h1len-1, h1len-1:]  = secondPart 

    #second matrix
    trans[h1len-1:, h1len-1:] = hmm2['transmat']

    #means
    means  = np.vstack((hmm1['means'], hmm2['means']))
    #covars
    covars = np.vstack((hmm1['covars'], hmm2['covars']))
    
    hmm3 = {}
    hmm3['startprob'] = pi
    hmm3['transmat']  = trans
    hmm3['means']     = means
    hmm3['covars']    = covars
    return hmm3


# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    alpha = np.zeros_like(log_emlik)
   
    alpha[0] = log_startprob[:-1] + log_emlik[0]
    #print(alpha[0])

    for i in range(1, log_emlik.shape[0]):
        for j in range(log_emlik.shape[1]):
            arr = alpha[i-1] + log_transmat[:-1,j]
            alpha[i,j] = logsumexp(arr) + log_emlik[i,j]
    return alpha
def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    beta = np.zeros_like(log_emlik)

    for i in range (log_emlik.shape[0]-2, -1,-1):
        for j in range(log_emlik.shape[1]):
            arr = log_transmat[j,:-1] + beta[i+1] + log_emlik[i+1]
            #print(arr)
            beta[i,j] = logsumexp(arr)
    return beta

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    probState = np.zeros_like(log_emlik)
    prevMax = np.zeros_like(log_emlik, dtype = np.int8)

    probState[0] = log_startprob[:-1] + log_emlik[0]

    for i in range(1, log_emlik.shape[0]):
        for j in range(log_emlik.shape[1]):
            arr = probState[i-1] + log_transmat[:-1,j]
            probState[i,j] = np.max(arr) + log_emlik[i,j]
            prevMax[i,j] = np.argmax(arr)

    path = []
    probPath =  probState[-1,-1]#np.max(probState[-1,:])
    currentInd = probState.shape[1] -1#np.argmax(probState[-1,:])
    print(currentInd)
    for i in range(log_emlik.shape[0]-1, -1,-1):
        path.insert(0, currentInd)
        currentInd = prevMax[i,currentInd]
    #print(prevMax)
    #print(probState[-1])
    return probPath, np.array(path)





def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    

    log_gamma = log_alpha + log_beta
    
    logExpSum = logsumexp(log_alpha[-1])
    log_gamma = log_gamma - logExpSum
    return log_gamma
    

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    #means = np.zeros((log_gamma.shape[1], 1))
    gamma = np.exp(log_gamma)
    means = gamma[0][np.newaxis].T * X[0]
    for i in range(1, X.shape[0]):
        tmp = gamma[i][np.newaxis].T * X[i]
        #print("gamma: ", gamma[i])
        #print(tmp)
        #print("\n")
        means += tmp
        

    sumGamma = np.sum(gamma, axis = 0)

    
    means = means/sumGamma[np.newaxis].T
    #print(means.shape)

    #covars
    covars = np.zeros_like(means)
    for i in range(log_gamma.shape[1]):
        numerator = np.sum(gamma[:,i][np.newaxis].T * np.square( X - means[i]) , axis = 0 )
        denominator = np.sum(gamma[:, i])
        #print("num", numerator)
        #print("den", denominator)
        covars[i] = numerator/denominator
    
    covars[covars < varianceFloor] = varianceFloor
    return means, covars
