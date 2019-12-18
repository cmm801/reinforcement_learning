import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple

ExponentialUtilityFunction = lambda ptf_val : -np.exp(-2*ptf_val)
ParameterRanges = namedtuple( 'ParamRange', [ 'low', 'high' ] )
class AssetProcess(ABC):
    
    def __init__(self, np_random, n_risky_assets=1, n_periods_per_year=52):
        self.np_random = np_random
        self.n_risky_assets = n_risky_assets
        self.n_periods_per_year = n_periods_per_year
        self.risk_free_rate = 0.01
        self.distrib = None
        self.reset_distribution()
    
    def generate_random_returns(self, n_samples=1):
        exc_rtns = self.distrib.random(n_samples=n_samples)
        ann_asset_rtns = self.risk_free_rate + np.hstack( [ np.zeros( (exc_rtns.shape[0],1) ), exc_rtns])
        return self._deannualize_return(ann_asset_rtns)
             
    def evolve(self):
        """ Given an asset distribution, this method evolves the process to the next time step.
            This method must be implemented for non-static processes, 
                but can be left as-is for static processes."""
        pass
        
    def reset_distribution(self):
        param_ranges = self.get_parameter_ranges()
        array_of_params = self.np_random.uniform( param_ranges.low, param_ranges.high )
        params = self.array_to_params(array_of_params)
        self.set_distribution(params)

    def _deannualize_return(self, rtn):
        return -1 + np.power(1 + rtn, 1/self.n_periods_per_year)
        
    @abstractmethod        
    def get_parameter_ranges(self):
        pass
    
    @abstractmethod
    def get_parameters(self):
        pass
    
    @abstractmethod
    def array_to_params(self, array):
        pass
    
    @abstractmethod
    def params_to_array(self, params):
        pass
    
    @abstractmethod
    def set_distribution(self, params):
        pass
    
class NormalStaticProcess(AssetProcess):
    
    def get_parameter_ranges(self):
        min_rtn, max_rtn = -0.05, +0.10
        min_std, max_std = 0.01, 0.30
        N = self.n_risky_assets
        mu_low  = min_rtn * np.ones( (N,) )
        mu_high = max_rtn * np.ones( (N,) )
        
        # Get the upper/lower bounds for covariance entries
        idx = np.tril_indices(N)        
        sigma_low_mtx = -(max_std ** 2) * np.ones((N,N), dtype=np.float32)
        sigma_low_mtx[np.eye(N) == 1] = min_std ** 2 # Variances must be non-negative
        sigma_low = sigma_low_mtx[idx].ravel()
        sigma_high = (max_std ** 2) * np.ones_like(sigma_low, dtype=np.float32)
        return ParameterRanges( low=np.hstack( [ mu_low, sigma_low ] ), 
                                high=np.hstack( [ mu_high, sigma_high ] ) )
        
    def get_parameters(self):
        return dict(mu=self.distrib.mu, sigma=self.distrib.sigma)
        
    def array_to_params(self, array):
        N = self.n_risky_assets
        assert len(array[N:]) == N*(N+1) // 2, 'Dimensional mismatch.'
        mu = array[:N]
        idx = np.tril_indices(N)
        chol = np.zeros((N,N))
        chol[idx] = array[N:]
        sigma = np.matmul(chol, chol.T)
        return dict( mu=mu, sigma=sigma )
    
    def params_to_array(self, params):
        chol = np.linalg.cholesky(params['sigma'])
        idx = np.tril_indices(self.n_risky_assets)
        return np.hstack( [ params['mu'].ravel(), chol[idx].ravel() ] )
        
    def set_distribution(self, params):
        self.distrib = NormalDistribution( self.np_random, params['mu'], params['sigma'])
            
class LognormalStaticProcess(NormalStaticProcess):
    def set_distribution(self, params):
        self.distrib = LognormalDistribution( self.np_random, params['mu'], params['sigma'])

class Distribution(ABC):

    def __init__(self, np_random ):
        self.np_random = np_random

    @abstractmethod
    def random(self, n_samples=1):
        pass
        
class NormalDistribution(Distribution):
        
    def __init__(self, np_random, mu, sigma ):
        super().__init__(np_random)        
        M = np.array(mu)
        S = np.array( sigma )
        assert S.ndim == 2 and S.shape[0] == S.shape[1], 'Covariance must be a square matrix.'
        assert M.ndim == 1 and S.shape[0] == M.size, 'Covariance and mean dimensions must be the same.'
        self.mu = M        
        self.sigma = S
        
    def random(self, n_samples=1):
        return self.np_random.multivariate_normal(self.mu, self.sigma, size=n_samples)
                
class LognormalDistribution(NormalDistribution):
    def random(self, n_samples=1):
        return -1 + np.exp( self.np_random.multivariate_normal(self.mu, self.sigma, size=n_samples) )
