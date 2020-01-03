import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from abc import ABC, abstractmethod, abstractstaticmethod
        
RT_COMPACT = 'compact'
RT_UNKNOWN = 'unknown'
RT_DISCRETE = 'discrete'
RT_UNBOUNDED = 'unbounded'
    
class Distribution(ABC):
    def __init__(self):
        self._set_name()

    @abstractmethod
    def prob(self, values):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' ) 
                
    def log_prob(self, values):
        p = self.prob(values)
        return tf.math.log(p)
    
    def set_distribution_from_params(self, params):
        self.set_distribution(**params)
                
    def _check_array_dimensions(self, array):
        assert array.shape[1] == self.get_number_of_params(), 'Parameter array has incorrect dimensions.'
            
    @abstractmethod
    def _check_param_dimensions(self, params):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' )             
    
    @abstractmethod
    def sample(self, n_samples=1):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' ) 

    @abstractmethod
    def entropy(self):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' ) 

    @abstractmethod    
    def _set_name(self):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' ) 
    
    @abstractmethod
    def set_distribution(self, **kwargs):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' )         

    @abstractmethod
    def get_number_of_params(self):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' ) 
    
    @abstractmethod
    def get_params_from_array(self, array):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' ) 
    
    @abstractmethod
    def get_array_from_params(self, params):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' )     
        
    def _preprocess_params(self, params):
        """Preprocess the parameters before setting them to define the distribution."""
        return params

    def set_distribution_from_array(self, array, preprocess=True):
        params = self.get_params_from_array(array)
        if preprocess:
            params = self._preprocess_params(params)
        self.set_distribution_from_params(params)


class Categorical(Distribution):
    def __init__(self, n_categories):
        super().__init__()
        self.dim = 1
        self.n_categories = n_categories
        self.range_type = RT_DISCRETE
    
    def prob(self, values):
        squeezed_vals = tf.squeeze( values, axis=2 )
        return self._distrib.prob(squeezed_vals)
    
    def entropy(self):
        return self._distrib.entropy()
    
    def sample(self, n_samples=1):
        raw_samples = self._distrib.sample(n_samples)
        return tf.expand_dims( raw_samples, axis=2 )
    
    def set_distribution(self, logits):
        self._check_param_dimensions(logits)
        self._distrib = tfp.distributions.Categorical(logits=logits)
        
    def get_number_of_params(self):
        return self.n_categories
    
    def get_params_from_array(self, array):
        self._check_array_dimensions(array)
        return dict(logits=array)
    
    def get_array_from_params(self, params):
        return params['logits']
    
    def _check_param_dimensions(self, logits):
        assert logits.shape[1] == self.n_categories, 'Parameter dimension does not match that of the distribution.'
        
    def _set_name(self):
        self.name = 'categorical'


class BaseContinuous(Distribution):
    def __init__(self, low, high):
        super().__init__()
        self.dim = low.size
        self._set_distribution_range(low=low, high=high)
        self._set_name()
        
    def prob(self, values):
        rescaled_values = self._rescale_to_range(values)
        return self._prob_rescaled(rescaled_values)            
    
    def sample(self, n_samples=1):
        unscaled_samples = self._sample_unscaled(n_samples=n_samples)
        return self._scale_samples_to_range(unscaled_samples)        
        
    def _set_distribution_range(self, low, high ):
        self.low = low
        self.high = high
        if np.any(low == -np.inf) or np.any(high == np.inf):
            self.range_type = RT_UNBOUNDED
        elif np.all(low > -np.inf) and np.all(high < np.inf):
            self.range_type = RT_COMPACT
        else:
            self.range_type = RT_UNKNOWN
        
    def _scale_samples_to_range(self, samples):
        if self.range_type == RT_UNBOUNDED:
            return np.clip( samples, self.low, self.high )
        elif self.range_type == RT_COMPACT:
            return self.low + (self.high - self.low) * samples
        else:
            raise ValueError( 'Unsupported range type: {}'.format(self.range_type) )
    
    def _rescale_to_range(self, values):
        if self.range_type == RT_UNBOUNDED:
            return values
        elif self.range_type == RT_COMPACT:
            return (values - self.low) / (self.high - self.low)
        else:
            raise ValueError( 'Unsupported range type: {}'.format(self.range_type) )

    @abstractmethod
    def _sample_unscaled(self, n_samples=1):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' ) 

    @abstractmethod
    def _prob_rescaled(self, rescaled_values):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' )


class TFPTFPContinuous(BaseContinuous):     
    def entropy(self):
        return self._distrib.entropy()
    
    def _prob_rescaled(self, value):
        #return tf.squeeze( self._distrib.prob(value), axis=2 )
        return self._distrib.prob(value)
    
    def _sample_unscaled(self, n_samples=1):
        return self._distrib.sample(n_samples)


class NormalDiag(TFPTFPContinuous):
    def set_distribution(self, loc, scale_diag):
        self._check_param_dimensions(loc)               
        params = dict(loc=loc, scale_diag=scale_diag)
        self._distrib = tfp.distributions.MultivariateNormalDiag(**params)
    
    def get_number_of_params(self):
        return 2 * self.dim
    
    def get_params_from_array(self, array):
        self._check_array_dimensions(array)        
        return dict( loc=array[:,:self.dim], \
                     scale_diag=np.vstack( array[:,self.dim:] ) )

    def get_array_from_params(self, params):
        return np.hstack( [ params['loc'], 
                            params[ 'scale_diag' ] ] )

    def _set_name(self):
        self.name = 'normal-diag'

    def _preprocess_params(self, params):
        params['scale_diag'] = tf.nn.relu( params['scale_diag'] )
        return params
    
    def _check_param_dimensions(self, loc):
        assert loc.shape[1] == self.dim, 'Parameter dimension does not match that of the distribution.'


class NormalFullCovariance(TFPTFPContinuous):
    def set_distribution(self,  loc, covariance_matrix ):
        self._check_param_dimensions(loc)        
        params = dict(loc=loc, covariance_matrix=covariance_matrix)
        self._distrib = tfp.distributions.MultivariateNormalFullCovariance(**params)
    
    def get_number_of_params(self):
        return self.dim * (self.dim + 1 )
    
    def get_params_from_array(self, array):
        self._check_array_dimensions(array)        
        return dict(loc=array[:,:self.dim], \
                    scale_diag=tf.linalg.diag( np.vstack(array[:,self.dim:]) ) )
    
    def get_array_from_params(self, params):
        cov = params['covariance_matrix']
        return np.hstack( [ params['loc'], 
                            tf.reshape(cov, (cov.shape[0], -1) ) ] )
    
    def _set_name(self):
        self.name = 'normal-full'              
    
    def _check_param_dimensions(self, loc):
        assert loc.shape[1] == self.dim, 'Parameter dimension does not match that of the distribution.'
        

class NormalTriL(TFPTFPContinuous):                    
    def set_distribution(self, loc, scale_tril ):
        self._check_param_dimensions(loc)               
        params = dict(loc=loc, scale_tril=scale_tril)
        self._distrib = tfp.distributions.MultivariateNormalTriL(**params)
    
    def get_number_of_params(self):
        return self.dim * (self.dim + 3) // 2
    
    def get_params_from_array(self, array):
        self._check_array_dimensions(array)        
        return dict(loc=array[:,:self.dim], \
                    scale_tril=tfp.math.fill_triangular(array[:,self.dim:] ) )
    
    def get_array_from_params(self, params):
        TriL = params['scale_tril']
        return np.hstack( [ params['loc'], 
                            tf.linalg.tfp.math.fill_triangular_inverse(TriL) ] )
    
    def _set_name(self):
        self.name = 'normal-tril'
    
    def _check_param_dimensions(self, loc):
        assert loc.shape[1] == self.dim, 'Parameter dimension does not match that of the distribution.'        


class Beta(TFPTFPContinuous):
    def entropy(self):
        return tf.squeeze( self._distrib.entropy(), axis=1 )
    
    def _prob_rescaled(self, value):
        return tf.squeeze( self._distrib.prob(value), axis=2 )
        
    def set_distribution(self, concentration1, concentration0):
        self._check_param_dimensions(concentration1, concentration0)
        params = dict(concentration1=concentration1, concentration0=concentration0)
        self._distrib = tfp.distributions.Beta(**params)
    
    def get_number_of_params(self):
        return 2
    
    def get_params_from_array(self, array):
        self._check_array_dimensions(array)        
        return dict( concentration1=array[:,[0]], \
                     concentration0=array[:,[1]] )
    
    def get_array_from_params(self, params):
        return np.hstack( [ params['concentration1'], params['concentration0'] ] )
    
    def _set_name(self):
        self.name = 'beta'
                    
    def _preprocess_params(self, params):
        params['concentration1'] = 1 + tf.nn.relu( params['concentration1'] )
        params['concentration0'] = 1 + tf.nn.relu( params['concentration0'] )
        return params
    
    def _check_param_dimensions(self, concentration1, concentration0):
        assert concentration1.shape[1] == self.dim, 'Parameter dimension does not match that of the distribution.'
        assert concentration0.shape[1] == self.dim, 'Parameter dimension does not match that of the distribution.'        
        assert concentration0.shape[0] == concentration1.shape[0], 'Inconsistent parameter dimensions.'
        

class Dirichlet(TFPTFPContinuous):
    def set_distribution(self, concentration ):
        self._check_param_dimensions(concentration)
        self._distrib = tfp.distributions.Dirichlet(concentration=concentration)
    
    def get_number_of_params(self):
        return self.dim
    
    def get_params_from_array(self, array):
        self._check_array_dimensions(array)        
        return dict(concentration=array)
    
    def get_array_from_params(self, params):
        return params['concentration']
    
    def _set_name(self):
        self.name = 'dirichlet'
                    
    def _preprocess_params(self, params):
        params['concentration'] = 1 + tf.nn.relu( params['concentration'] )
        return params
    
    def _check_param_dimensions(self, concentration):
        assert concentration.shape[1] == self.dim, 'Parameter dimension does not match that of the distribution.'
        

class MultiBeta(TFPTFPContinuous):
    def set_distribution(self, concentration1, concentration0 ):
        concentration1 = np.array(concentration1, dtype=np.float32)
        concentration0 = np.array(concentration0, dtype=np.float32)
        self._distrib = [ Beta(c_1, c_0) for c_1, c_0 in zip(concentration1, concentration0) ]
        
    def prob(self, value):
        value = np.atleast_2d(value)
        independent_probs = [ self._distrib[j].prob(value[:,j]) for j in range(self.n_dims) ]
        return np.product( independent_probs, axis=0 )
    
    def entropy(self):
        return np.sum( [ d.entropy() for d in self._distrib ] )
    
    def _sample_unscaled(self, n_samples=1):
        samples = np.hstack( [ d.sample(n_samples) for d in self._distrib ] )
        
    def get_number_of_params(self):
        return 2 * self.dim
    
    def get_params_from_array(self, array):
        self._check_array_dimensions(array)        
        return dict(concentration1=array[:,:0], \
                    concentration0=array[:,:1] )
    
    def get_array_from_params(self, params):
        return np.hstack( [ params['concentration1'], params['concentration0'] ] )
    
    def _set_name(self):
        self.name = 'multi-beta'
                    
    def _preprocess_params(self, params):
        params['concentration1'] = 1 + tf.nn.relu( params['concentration1'] )
        params['concentration0'] = 1 + tf.nn.relu( params['concentration0'] )
        return params

