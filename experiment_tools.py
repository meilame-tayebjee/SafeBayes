import numpy as np
from scipy.special import psi #Digamma function useful for RLogLoss
from scipy.stats import invgamma

#Correct vs wrong model data generation
def generate_correct_model(n , sigma_sq = 1/40, pmax = 50):
    sigma = np.sqrt(sigma_sq)
    X = np.random.randn(n, pmax)
    X = np.hstack([np.ones((n, 1)), X]) # add intercept
    eps = sigma * np.random.randn(n)

    beta = np.hstack([ [0], np.ones(4)/10, np.zeros(pmax - 4)]) #as described in the paper: true beta is (0,0.1,...,0.1,0,...,0)

    y = X@beta + eps

    return X, y, beta


def generate_wrong_model(n , sigma_sq = 1/20, pmax = 50):
    coin_flip = (np.random.rand(n) > 0.5).astype(bool) #choose "relevant" indexes
    sigma = np.sqrt(sigma_sq)


    X = np.zeros((n, pmax))
    X[coin_flip, :] = np.random.randn(n, pmax)[coin_flip, :] #only for the relevant indexes, we draw from N(0,1)

    X = np.hstack([np.ones((n, 1)), X]) # add intercept

    eps = sigma*np.random.randn(coin_flip.sum()) #only for the relevant indexes

    beta = np.hstack([ [0], np.ones(4)/10 , np.zeros(pmax - 4)]) #as described in the paper: true beta is (0,0.1,...,0.1,0,...,0)

    #draw y
    y = np.zeros(n)
    y[coin_flip] = X[coin_flip]@beta + eps #correct model for the relevant indexes, 0 otherwise

    return X, y, beta

#Functions to calculate the generalized posterior distributions
#See sec 2.5 of the paper/ sec 3.1 of the longer paper

#------------BETA----------------#
#Covariance matrix of the multivariate normal **posterior** for beta
# Remark : covariance matrix is in fact sigma_sq * Sigman_eta
def Sigman_eta(Sigma, Xn, eta):
  return np.linalg.inv(np.linalg.inv(Sigma) + eta * np.transpose(Xn) @ Xn)
#Expectation of the multivariate normal **posterior** for beta
def betan_eta(Sigma, Xn, eta, beta0, yn):
  return Sigman_eta(Sigma, Xn, eta) @ (np.linalg.inv(Sigma) @ beta0 + eta * np.transpose(Xn) @ yn)


#================================#

#------------SIGMA_SQ----------------#
#Parameters of the Inv-Gamma **posterior** for sigma_sq
def an_eta(a0, eta, n):
  return a0 + (eta * n)/2
def bn_eta(b0, eta, beta, Sigma, Xn, yn):
  betan = betan_eta(Sigma, Xn, eta, beta, yn)
  # sigman = Sigman_eta(sigma, Xn, eta)
  # return b0 + 1/2 * beta.T @ np.linalg.inv(sigma) @ beta + eta/2 * yn.T @ yn -1/2 * betan.T @ np.linalg.inv(sigman) @ betan
  return b0 + (eta / 2) * np.sum(np.square(yn - (Xn @ betan)))
#The induced expectation
def sigman_eta2(a,b):
  return b/(a-1)
#================================#


# Functions defined for varying sigma^2 section 4.2 (bigger paper)

def RLogLoss(Xi, yi, eta, i, beta0, a0, b0, Sigma_0):
  if i>=1:
    a = an_eta(a0,eta,i)
    b = bn_eta(b0, eta, beta0, Sigma_0, Xi[:-1], yi[:-1])
    beta = betan_eta(Sigma_0, Xi[:-1], eta, beta0, yi[:-1])
    Sigma = Sigman_eta(Sigma_0, Xi[:-1], eta)
  else:
    a = a0
    b = b0
    beta = beta0
    Sigma = Sigma_0
  return 1/2 * np.log(2 * np.pi * b) - 1/2* psi(a) + 1/2 * (yi[-1] - Xi[-1] @ beta)**2 / (b/a) + 1/2 * Xi[-1].T @ Sigma @ Xi[-1]


def ILogLoss(Xi, yi, eta, i, beta0, a0, b0, Sigma_0):
  if i>=1:
    a = an_eta(a0,eta,i)
    b = bn_eta(b0, eta, beta0, Sigma_0, Xi[:-1], yi[:-1])
    beta = betan_eta(Sigma_0, Xi[:-1], eta, beta0, yi[:-1])
    sigma2 = sigman_eta2(a,b)
  else:
    a = a0
    b = b0
    beta = beta0
    sigma2 = sigman_eta2(a,b)
  return 1/2 * np.log(2 * np.pi * sigma2) - psi(a) + 1/2 * (yi[-1] - Xi[-1] @ beta)**2 / (sigma2)

def SafeBayes(X, y, step_size=1, k_max=16, loss = RLogLoss, pmax=50, sigma2=1/40):
  num = int(k_max/step_size)
  etas = np.linspace(0,k_max,num)
  S_etas = np.zeros(num)

  # Constants to get the algorithm going
  # Found section A.2 of the paper -> this is not the good experiment, look at section 5.4 (long article) where they say they use same constant as in 5.1
  beta0 = np.zeros(pmax+1)
  a0 = 1
  b0 = sigma2 * a0
  #Sigma_0 = 10e3 * np.eye(pmax+1)
  Sigma_0 = np.eye(pmax+1)

  for eta, k in zip(etas, range(num)):
    for i in range(1,len(X)):
      Xi = X[:(i+1)]
      yi = y[:(i+1)]
      S_etas[k] += loss(Xi, yi, 2**(-eta), i, beta0, a0, b0, Sigma_0)
  eta = etas[np.argmin(S_etas)]
  return 2**(-eta)


def square_risk(beta,X_test,y_test):
    return np.mean((y_test - X_test@beta)**2)

def experiment(X, y,X_test,y_test, step_size=1, k_max=16, pmax=50, sigma2=1/40):
  num = int(k_max/step_size)
  N = len(X)
  etas = np.linspace(0,k_max,num)
  S_R_Log_etas = np.zeros((N-1,num))
  S_I_Log_etas = np.zeros((N-1,num))

  # Constants to get the algorithm going
  beta0 = np.zeros(pmax+1)
  a0 = 1
  b0 = sigma2 * a0
  Sigma_0 = np.eye(pmax+1)

  for eta, k in zip(etas, range(num)):
    for i in range(1,len(X)):
      Xi = X[:(i+1)]
      yi = y[:(i+1)]
      S_R_Log_etas[(i-1):,k] += RLogLoss(Xi, yi, 2**(-eta), i, beta0, a0, b0, Sigma_0)
      S_I_Log_etas[(i-1):,k] += ILogLoss(Xi, yi, 2**(-eta), i, beta0, a0, b0, Sigma_0)


  best_etas_R_Log = etas[np.argmin(S_R_Log_etas,axis=1)]
  best_etas_I_Log = etas[np.argmin(S_I_Log_etas,axis=1)]
  square_risks = np.empty((N-1, 3))
  for k in range(N-1):
    beta_bayes = betan_eta(Sigma_0, X[:k], 1, beta0, y[:k])
    beta_R_Log_safe_bayes = betan_eta(Sigma_0, X[:k], 2**(-best_etas_R_Log[k]), beta0, y[:k])
    beta_I_Log_safe_bayes = betan_eta(Sigma_0, X[:k], 2**(-best_etas_I_Log[k]), beta0, y[:k])
    square_risks[k] = [square_risk(beta_bayes,X_test,y_test),square_risk(beta_R_Log_safe_bayes,X_test,y_test),square_risk(beta_I_Log_safe_bayes,X_test,y_test)]

  return square_risks