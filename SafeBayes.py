import numpy as np
import time
from scipy.special import psi, softmax #Digamma function useful for RLogLoss
from scipy.stats import invgamma


#================================#
class SafeBayesLinearRegression():

    def __init__(self, loss = 'RLogLoss', a0 = 1, b0 = 1/40,  step_size=1, k_max=16, pmax=50, beta_0 = None, Sigma_0 = None):
        """
        Parameters:
        -----------
        loss : str, default = 'RLogLoss'
            Loss function to use. Either 'RLogLoss' or 'ILogLoss'
        a0 : float, default = 1
            Prior shape parameter for sigma_sq
        b0 : float, default = 1/40
            Prior scale parameter for sigma_sq
        step_size : int, default = 1
            Step size for SafeBayes algorithm
        k_max : int, default = 16
            Maximum value for SafeBayes algorithm
        pmax : int, default = 50
            Maximum number of features
        beta_0 : numpy.array, default = None
            Prior mean for beta. If None, will be set to 0
        Sigma_0 : numpy.array, default = None
            Prior covariance matrix for beta. If None, will be set to the identity matrix
        """
        if loss not in ['RLogLoss', 'ILogLoss']:
            raise ValueError('loss should be either "RLogLoss" or "ILogLoss"')

        #loss function
        if loss == 'RLogLoss':
            self.loss = self.RLogLoss
        else:
            self.loss = self.ILogLoss

        #prior parameters
        self.pmax = pmax

        if beta_0 is not None:
            if beta_0.shape[0] != pmax+1:
                raise ValueError('beta_0 should have the same length as the number of features in X')
            self.beta_0 = beta_0
        else:
            self.beta_0 = np.zeros(pmax+1) #prior mean for beta

        if Sigma_0 is not None:
            if Sigma_0.shape[0] != pmax+1 or Sigma_0.shape[1] != pmax+1:
                raise ValueError('Sigma_0 should be a square matrix with the same length as the number of features in X')
            self.Sigma_0 = Sigma_0
        else:
            self.Sigma_0 = np.eye(pmax+1)

        self.a0 = a0 #prior shape parameter for sigma_sq
        self.b0 = b0 #prior scale parameter for sigma_sq


        #hyperparameters for SafeBayes algorithm
        self.step_size = step_size 
        self.k_max = k_max


        #eta
        self.eta = 1 #default value

        #posterior
        self.beta_params = None
        self.sigma_sq_params = None

    
    #------------BETA----------------#
    #Covariance matrix of the multivariate normal **posterior** for beta
    # Remark : covariance matrix is in fact sigma_sq * Sigman_eta
    def Sigman_eta(self, Xn, Sigma,eta):
        return np.linalg.inv(np.linalg.inv(Sigma) + eta * Xn.T @ Xn)

    #Expectation of the multivariate normal **posterior** for beta

    def betan_eta(self, Xn, yn, Sigma, eta, beta0):
        return self.Sigman_eta(Xn, Sigma, eta) @ (np.linalg.inv(Sigma) @ beta0 + eta * Xn.T @ yn)

    #================================#

    #------------SIGMA_SQ----------------#
    #Parameters of the Inv-Gamma **posterior** for sigma_sq

    def an_eta(self, a0, eta, n):
        return a0 + (eta * n)/2


    def bn_eta(self, Xn, yn, b0, eta, mean_beta_posterior):
        return b0 + (eta / 2) * np.sum(np.square(yn - (Xn @ mean_beta_posterior)))


    def sigman_eta2(self, a,b):
        return b/(a-1)

# Functions defined for varying sigma^2 section 4.2 (bigger paper)

    def RLogLoss(self, Xi, yi, eta, i, beta0, a0, b0, Sigma_0):
        if i>=1:

            beta = self.betan_eta( Xi[:-1], yi[:-1], Sigma_0, eta, beta0)
            Sigma = self.Sigman_eta(Xi[:-1], Sigma_0, eta)
            a = self.an_eta(a0,eta,i)
            b = self.bn_eta(Xi[:-1], yi[:-1], b0, eta, beta)
        else:
            a = a0
            b = b0
            beta = beta0
            Sigma = Sigma_0
        return 1/2 * np.log(2 * np.pi * b) - 1/2* psi(a) + 1/2 * (yi[-1] - Xi[-1] @ beta)**2 / (b/a) + 1/2 * Xi[-1].T @ Sigma @ Xi[-1]

    def ILogLoss(self, Xi, yi, eta, i, beta0, a0, b0, Sigma_0):
        if i>=1:

            beta = self.betan_eta( Xi[:-1], yi[:-1], Sigma_0, eta, beta0)
            a = self.an_eta(a0,eta,i)
            b = self.bn_eta(Xi[:-1], yi[:-1], b0, eta, beta)
            sigma2 = self.sigman_eta2(a,b)
        else:
            a = a0
            b = b0
            beta = beta0
            sigma2 = self.sigman_eta2(a,b)
        return 1/2 * np.log(2 * np.pi * sigma2) - psi(a) + 1/2 * (yi[-1] - Xi[-1] @ beta)**2 / (sigma2)


    
    def SafeBayes(self, X, y):
        num = int(self.k_max/self.step_size)
        etas = np.linspace(0,self.k_max,num)
        S_etas = np.zeros(num)

        # Constants to get the algorithm going
        # Found section A.2 of the paper -> this is not the good experiment, look at section 5.4 (long article) where they say they use same constant as in 5.1
        beta0 = self.beta_0
        a0 = self.a0
        b0 = self.b0
        #Sigma_0 = 10e3 * np.eye(pmax+1)
        Sigma_0 = np.eye(self.pmax+1)

        for eta, k in zip(etas, range(num)):
            for i in range(1,len(X)):
                Xi = X[:(i+1)]
                yi = y[:(i+1)]
                S_etas[k] += self.loss(Xi, yi, 2**(-eta), i, beta0, a0, b0, Sigma_0)
        eta = etas[np.argmin(S_etas)]
        self.eta = 2**(-eta)
        return self.eta
    
    def GeneralizedPosteriors(self, X, y, eta):

        """ For a given eta, return the generalized posteriors parameters for beta and sigma_sq
            Note that for eta = 1, this gives the "classic" posteriors

            Params:
            ----------
            X,y : data (numpy.arrays, shape (n x p+1), (n,1)
            eta : learning rate for generalized posterior

            beta_0, Sigma_0 : prior parameters for the beta prior multivariate normal distribution
                NOTE : actually, the distribution of beta depends on sigma_sq and the true variance is sigma_sq * Sigma_0

            a0, b0 : prior parameters for the inverse gamma distribution of sigma_sq


            Returns:
            ----------
            beta_params: tuple containing post. multi. normal mean and Sigma_n_eta (variance is once again sigma_sq * Sigma_n_eta)
            sigma_sq_params : tuple containing  posterior inverse-gamma dist. mean and parameters for sigma_sq
        """

        n= y.shape[0]

        ####----Compute Beta posterior params-------##
        Sigma_n_eta = self.Sigman_eta(X, self.Sigma_0, eta)
        mean_beta_posterior = self.betan_eta(X, y, self.Sigma_0, self.eta, self.beta_0)

        beta_params = (mean_beta_posterior, Sigma_n_eta)

        ####----Compute sigma_sq posterior-----#
        a_posterior = self.an_eta(self.a0, eta, n)
        b_posterior = self.bn_eta(X, y, self.b0, eta, mean_beta_posterior)
        mean_sigma_sq_posterior = b_posterior / (a_posterior - 1)

        sigma_sq_params = (mean_sigma_sq_posterior, a_posterior, b_posterior)

        self.beta_params = beta_params
        self.sigma_sq_params = sigma_sq_params


        return beta_params, sigma_sq_params
    


    def fit(self, X,y, verbose = False):

        """ Wrapper for the full pipeline:
            1. Run Safe Bayes for finding optimal eta
            2. Return the Generalized posteriors for this optimal eta

            Verbose (bool) for displaying
        """


        n = y.shape[0]
        if verbose:
            print("Running Safe Bayes to find optimal eta...")

        start = time.time()
        eta = self.SafeBayes(X, y)
        end = time.time()
        if verbose:
            print("Found eta* = {} in {} s".format(round(eta, 2), round(end-start, 2)))



        if verbose:
            print("Computing generalized posteriors...")

        beta_params, sigma_sq_params = self.GeneralizedPosteriors(X, y, eta)

        if verbose:
            print("Done.")

        return beta_params, sigma_sq_params, eta
    

    

