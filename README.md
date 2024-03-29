# Safe Bayes Linear Regression
## Eliot Beyler, Sébastien Melo and Meilame Tayebjee | MVA 2023-2024

This project has been realised for the Bayesian Machine Learning course of the MVA master by Prof. R. Bardenet and Prof. J. Arbel.

We propose an implementation of the Safe Bayes Linear Regression from the article _Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It_ by P. Grünwald & T. van Ommen (2017), as well as experiments. The SafeBayes Linear Regression is a regularized Bayesian Linear Regression that is robust to model misspecification (hetereoscedasticity, corrupted data, etc.).


The module ```SafeBayes.py``` is ready to use:
- Optional: Run ``` pip install -r requirements.txt ```   to install the required packages (nothing fancy, only numpy and scipy)*
- Use your data ```X, y```
- Instantiate a ```SafeBayesLinearRegression()``` object. Default prior parameters from the article are already implemented for you, but you can customize them !
- Calling the ```.fit(X, y)``` method will directly run the SafeBayes algorithm to find the optimal $\eta$ and yield the $\eta$-generalized posterior distributions parameters (that is mean and std of the multivariate normal for $\beta$ and shape/scale of the inverse gamma for $\sigma^2$).


If you wish not to run the SafeBayes algorithm, you can also call the ```.GeneralizedPosteriors(X, y, eta)``` method with the $\eta$ of your choice! Note that $\eta=1$ will yield the standard Bayesian Linear Regression.

Feel free to reach out to us at eliot.beyler@ens.psl.eu / sebastien.melo@polytechnique.edu / meilame.tayebjee@polytechnique.edu for any questions or recommendations !


*Remark : to run the experiments, you also need to install matplotlib, pandas and tqdm.