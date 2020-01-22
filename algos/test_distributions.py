import numpy as np
import tensorflow as tf
import gym
from abc import ABC, abstractmethod, abstractstaticmethod
import distributions

def run_all_tests():    
    cat = CategoricalTests()
    cat.run_tests()
    
    norm_diag = NormalDiagTests()
    norm_diag.run_tests()
    
    norm_tril = NormalTriLTests()
    norm_tril.run_tests()    
    
    beta = BetaTests()
    beta.run_tests()
    
    dirichet = DirichletTests()
    dirichet.run_tests()
    
    multibeta = MultiBetaTests()
    multibeta.run_tests()
    
    print( 'All tests have passed.')

class BaseTest(ABC):

    def __init__(self):
        self.ran_gen = np.random.RandomState(121)
        
    def run_tests(self):        
        self._test_sampling()    
        self._test_prob()
        self._test_params_extraction()
        self._test_entropy()
        print( 'Tests have passed for {}'.format(self.__class__.__name__) )
    
    def _test_sampling(self):
        distribs = self.get_distributions()
        for dist in distribs:
            n_obs = self.ran_gen.choice(20)
            n_samples = self.ran_gen.choice(20)
            n_params = dist.get_number_of_params()            
            array = self.ran_gen.randn(n_obs, n_params)
            dist.set_distribution_from_array(array) 
            samples = dist.sample(n_samples)
            if dist.dim > 0:
                assert samples.shape == (n_samples, n_obs, dist.dim ), 'Incorrect dimensions.'
            else:
                assert samples.shape == (n_samples, n_obs ), 'Incorrect dimensions.'
    
    def _test_prob(self):
        distribs = self.get_distributions()
        for dist in distribs:
            n_obs = self.ran_gen.choice(20)
            n_params = dist.get_number_of_params()
            array = self.ran_gen.randn(n_obs, n_params)
            dist.set_distribution_from_array(array)
            n_samples = self.ran_gen.choice(20) 
            samples = dist.sample(n_samples=n_samples)
            assert dist.prob(samples).shape == (n_samples, n_obs), 'Probability shape is incorrect.'
            assert dist.log_prob(samples).shape == (n_samples, n_obs), 'Log Probability shape is incorrect.'
    
    def _test_params_extraction(self):
        """Check that the parameters are correctly extracted from arrays."""
        def __check_params_extraction( dist, array, n_params):
            try:
                dist.get_params_from_array(array)
            except AssertionError:
                if array.shape[1] == n_params:
                    raise Exception( 'Exception should not be raised when parameters match the distribution dimension.' )
            except:
                if array.shape[1] != n_params:
                    raise Exception( 'Exception should be raised when parameters do not match the distribution dimension.' )
                    
        distribs = self.get_distributions()
        for dist in distribs:
            n_obs = self.ran_gen.choice(20)
            n_samples = self.ran_gen.choice(20)
            n_params = dist.get_number_of_params()            
            for j in [ -1, 0, 1 ]:
                array = self.ran_gen.randn( n_obs, max( 1, n_params + j ) )
                __check_params_extraction( dist, array, n_params)        
    
    def _test_entropy(self):
        distribs = self.get_distributions()
        for dist in distribs:
            n_obs = self.ran_gen.choice(20)
            n_params = dist.get_number_of_params()
            array = tf.ones( (n_obs, n_params) )
            dist.set_distribution_from_array(array)
            assert dist.entropy().numpy().shape == (n_obs,), 'Entropy shape is incorrect.'

    @abstractmethod
    def get_distributions(self):
        raise NotImplementedError( 'Abstract method must be overridden by subclass.' )         

        
class CategoricalTests(BaseTest):
    def _test_sampling(self):
        super()._test_sampling()
        distribs = self.get_distributions()
        for dist in distribs:
            n_obs = self.ran_gen.choice(20)
            n_samples = self.ran_gen.choice(20)            
            n_params = dist.get_number_of_params()            
            array = self.ran_gen.randn(n_obs, n_params)
            dist.set_distribution_from_array(array) 
            samples = dist.sample(n_samples)
            allowable_values = set( np.arange(dist.n_categories) )
            sample_set = set( tf.reshape( samples, (-1,) ).numpy() )
            assert sample_set.issubset( allowable_values ), 'Categorical Samples are outside of allowable values.'

    def _test_prob(self):
        super()._test_prob()                                   
        distribs = self.get_distributions()
        for dist in distribs:
            n_obs = self.ran_gen.choice(20)
            n_samples = self.ran_gen.choice(20)            
            n_params = dist.get_number_of_params()            
            array = tf.ones( (n_obs, n_params) )
            dist.set_distribution_from_array(array)
            N = dist.n_categories
            for j in range(N):
                assert np.all( np.isclose( 1/N, dist.prob( tf.convert_to_tensor( np.atleast_3d(j) ) ) ) ), \
                                            'Probability values are incorrect.'
                assert np.all( np.isclose( np.log(1/N), dist.log_prob( np.atleast_3d(j) ) ) ), \
                                            'Log Probability values are incorrect.'

    def get_distributions(self):
        distribs = []
        for n_categories in [ 1, 2, 3, 6, 11 ]:
            distribs.append( distributions.Categorical(n_categories=n_categories) )
        return distribs
        

class BetaTests(BaseTest):
    def _test_prob(self):
        super()._test_prob()                                   
        distribs = self.get_distributions()
        for dist in distribs:
            n_obs = self.ran_gen.choice(20)
            n_samples = self.ran_gen.choice(20)            
            n_params = dist.get_number_of_params()            
            array = tf.ones( (n_obs, n_params) )
            dist.set_distribution_from_array(array)
            L = dist.low
            H = dist.high
            M = (L + H) / 2
            assert np.all( np.isclose( 0,   dist.prob( tf.convert_to_tensor( np.atleast_3d(L) ).numpy() ) ) ), \
                                                                'Probability values are incorrect.'
            assert np.all( np.isclose( 0,   dist.prob( tf.convert_to_tensor( np.atleast_3d(H) ).numpy() ) ) ), \
                                                                'Probability values are incorrect.'
            assert np.all( np.isclose( 1.5, dist.prob( tf.convert_to_tensor( np.atleast_3d(M) ).numpy() ) ) ), \
                                                                'Probability values are incorrect.'
            assert np.all( np.isclose( -np.inf,     dist.log_prob( tf.convert_to_tensor( np.atleast_3d(L) ).numpy() ) ) ), \
                                                                'Probability values are incorrect.'
            assert np.all( np.isclose( -np.inf,     dist.log_prob( tf.convert_to_tensor( np.atleast_3d(H) ).numpy() ) ) ), \
                                                                'Probability values are incorrect.'
            assert np.all( np.isclose( np.log(1.5), dist.log_prob( tf.convert_to_tensor( np.atleast_3d(M) ).numpy() ) ) ), \
                                                                'Probability values are incorrect.'

    def get_distributions(self):
        low_arr = [ 0, -10, 0, 4 ]
        high_arr = [ 1, 10, 12, 11 ]
        distribs = []
        for j in range(len(low_arr)):
            low = tf.convert_to_tensor([ low_arr[j] ])
            high = tf.convert_to_tensor( [high_arr[j]] )
            distribs.append( distributions.Beta( low=low, high=high ) )
        return distribs                                   


class BaseNormalTest(BaseTest):    
    pass


class NormalDiagTests(BaseNormalTest):
    def get_distributions(self):
        dims = [ 1, 2, 3, 8 ]
        distribs = []
        for dim in dims:
            low = self.ran_gen.randn(dim)
            length = 0.1 + self.ran_gen.choice(20, dim)
            high = low + length
            distribs.append( distributions.NormalDiag( low=low, high=high ) )
        return distribs      


class NormalTriLTests(BaseNormalTest):
    def get_distributions(self):
        dims = [ 1, 2, 5, 6 ]
        distribs = []
        for dim in dims:
            low = self.ran_gen.randn(dim)
            length = 0.1 + self.ran_gen.choice(20, dim)
            high = low + length
            distribs.append( distributions.NormalTriL( low=low, high=high ) )
        return distribs

    
class DirichletTests(BaseTest):
    def get_distributions(self):
        dims = [ 2, 4, 7, 10 ]
        distribs = []
        for dim in dims:
            low = self.ran_gen.randn(dim)
            length = 0.1 + self.ran_gen.choice(20, dim)
            high = low + length
            distribs.append( distributions.Dirichlet( low=low, high=high ) )
        return distribs
        
    
class MultiBetaTests(BaseTest):
    def get_distributions(self):
        dims = [ 1, 2, 5, 8 ]
        distribs = []
        for dim in dims:
            low = self.ran_gen.randn(dim)
            length = 0.1 + self.ran_gen.choice(20, dim)
            high = low + length
            distribs.append( distributions.MultiBeta( low=low, high=high ) )
        return distribs    