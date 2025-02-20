

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def EI(y_mean, std, y_max, xi=0):
    """Expected improvement acquisition function
    """
    z = (y_mean - y_max - xi) / std
    res = (y_mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
    return res


def UCB(y_mean, std, y_max=None, kappa=1):
    """Upper Confidence Bound acquition function"""
    return y_mean + kappa * std



class BayesOpt:
    """Bayesian Optimization class to use with GPref
    """
    def __init__(self, infer, bounds = None, acquisition="EI", opt_method = "L-BFGS-B"):
        self.method = opt_method
        self.bounds = bounds
        self.infer = infer
        
        # Currently does not use the exploration parameter in these functions (only uses the defaults)
        if acquisition == "EI":
            self.acquisition = EI
        elif acquisition == "UCB":
            self.acquisition = UCB
        else:
            raise ValueError("acquisition should be 'EI' or 'UCB'.")

        self.y_max = infer.fMAP.max()
        self.x_max_ind = infer.fMAP.argmax()
        self.x_max = infer.X[self.x_max_ind]
        self.x_dims = infer.X.shape[1]
        
        if bounds is None:
            raise ValueError("You need to define an array with (n_variables, 2) shape to define the upper and lower bounds of each variable.")
        if bounds.shape != (self.x_dims,2):
            raise ValueError("You need to define an array with (n_variables, 2) shape to define the upper and lower bounds of each variable.")


    def aqc_optim(self, x):
        """Objective function to optimize for acquisition"""
        mu, cov = self.infer.predict(x)
        y_mean = mu.ravel()
        std = np.sqrt(np.diag(cov))
        # acquisition either UCB or EI
        ys = self.acquisition(y_mean, std, self.y_max)
        return ys

    def propose_next(self, n_samp=1, n_solve=1):
        """Optimizes the acquisition function"""
        x_seeds, max_acq, x_max = self.initial_xtries(n_samp, n_solve)
        for x_try in x_seeds: 
            res = minimize(lambda x: -self.aqc_optim(x.reshape(1, -1)).ravel(), x_try, bounds=self.bounds, method=self.method)
            if not res.success:
                continue
            if max_acq is None or -res.fun >= max_acq:
                x_max = res.x
                max_acq = -res.fun
        return np.clip(x_max, self.bounds[:, 0], self.bounds[:, 1]).reshape(1, self.x_dims)

    def initial_xtries(self, n_samp, n_solve):
        """Create initial x_seeds to try for propose_next. Acquisition function is optimized for each of those.
        n_samp is the number of initial points sampled. n_solve is how many number of times it will be solved."""
        x_tries = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size =(n_samp, self.x_dims))
        ys = self.aqc_optim(x_tries)
        max_acq = ys.max()
        x_max = x_tries[ys.argmax()]
        x_seeds = x_tries[np.argsort(ys.flat)][:n_solve]
        return x_seeds, max_acq, x_max
    
    def create_grid(self, num_intervals=20):
        """Create grid of all possible input values"""
        intervals = np.linspace(self.bounds[:,0],self.bounds[:,1], num_intervals)
        return np.array(np.meshgrid(*np.split(intervals, self.x_dims, axis=1))).reshape(self.x_dims,-1).T
    
    def propose_exhaustive(self, num_intervals = 20):
        """Proposes next x value by exhaustive search in grid of all inputs"""
        grid = self.create_grid(num_intervals)
        acq_values = self.aqc_optim(grid)
        arg_max_acq = acq_values.argmax() 
        return grid[arg_max_acq]
       
    
def create_comparison_data(bounds, true_vector, n_samp=10):
    """Create comparison data 
    
    returns X = vectors values, index of compared (left one preferred)
    """
    n_vars = bounds.shape[0]

    comp_1 = np.random.uniform(bounds[:,0], bounds[:,1], size=(n_samp,n_vars))
    comp_2 = np.random.uniform(bounds[:,0], bounds[:,1], size=(n_samp,n_vars))
    left_pref = np.linalg.norm(comp_1 - np.array(true_vector),axis=1) < np.linalg.norm(comp_2 - np.array(true_vector),axis=1)
    X = np.vstack([comp_1, comp_2])

    preferred_index = np.where(left_pref,np.arange(0,n_samp), np.arange(n_samp,2*n_samp))
    unpreferred_index = np.where(left_pref,np.arange(n_samp,2*n_samp), np.arange(0,n_samp))
    Pairs = np.vstack([preferred_index, unpreferred_index]).T
    
    return X, Pairs