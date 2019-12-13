import gym
import numpy as np
from collections import namedtuple
import distributions

ObsParams = namedtuple( 'ObsParams', [ 'portfolio_value', 'weight', 'process_params' ] )

class PortfolioEnv(gym.Env):
    """ A portfolio construction environment for OpenAI gym. 
        action_space: vectors with entries between -1 and +1, representing changes
           to the current allocation (positive values <--> buying an asset, 
                                      negative values <--> selling an asset.)
        observation_space: states contain the portfolio value, and the
           assets' weights, means and Cholesky lower-diagonal matrix. 
    """
    metadata = {'render.modes': ['human']}
    
    def __init__( self, 
                  n_risky_assets=1,
                  benchmark=None,
                  utility_fun=None, 
                  trans_costs=0.0,
                  allow_leverage=False, 
                  allow_shorting=False,
                  trading_periods_per_year=52,
                  is_static=True,
                  asset_process_name='lognormal-static'
                ):
        """This environment takes the parameters of its distribution as inputs."""
        super( self.__class__, self).__init__()
    
        # Save input arguments
        self.asset_process_name = asset_process_name
        self.n_risky_assets = n_risky_assets
        self.trading_periods_per_year = trading_periods_per_year
        self.utility_fun = utility_fun

        # Initialize the asset process and other parameters
        self.reset()
        
        # Actions are the changes in weights of all assets (cash + the risky assets)
        N = n_risky_assets        
        self.action_space = gym.spaces.Box(
            low = -np.ones( (N+1,) ), 
            high = np.ones( (N+1,) ) 
        )

        # Define the dimensions of the observation space
        param_ranges = self.asset_process.get_parameter_ranges()
        min_ptf_val, max_ptf_val = 0, np.inf
        low  = ObsParams( portfolio_value=min_ptf_val, weight=-np.ones((N,)), process_params=param_ranges.low )
        high = ObsParams( portfolio_value=max_ptf_val, weight=+np.ones((N,)), process_params=param_ranges.high )
        self.observation_space = gym.spaces.Box(
            low  = np.hstack(list(low)),
            high = np.hstack(list(high)) )

                         
    def seed(self, seed=None):
        self.np_random, seed = gym.seeding.np_random(seed)
        return [seed]
    
    def step(self, action):        
        assert not np.any(np.isnan(action)), 'Actions must always be valid numbers.'

        # Get the covariance matrix before the transition to the next state
        old_ptf_val = self.portfolio_value
            
        # Get the new portfolio value
        simple_rtns = self._get_random_returns(self)

        # Get the new portfolio value
        chg_wts = np.hstack( [ 1 - sum(action), action ] )
        new_weights = np.clip( self.weights + action, 0, 1 )
        ptf_rtn = np.matmul( new_weights, new_simple_rtns.T )
        new_ptf_val = max( 0, self.ptf_val * ( 1 + ptf_rtn ) )

        # Evolve the asset process (if it is non-static)
        self.asset_process.evolve()
        
        # Calculate the reward from the old and new state parameters
        obs = self._get_observation( new_ptf_val, new_weights )
        reward = self._calculate_reward( action, old_ptf_val, new_ptf_val )
        done = (self.portfolio_value == 0)
               
        info = {'simple_rtns' : simple_rtns}        
        return obs, reward, done, info
        
    def reset(self):
        """Reset the state of the environment to an initial state."""        
        # Generate new parameters
        self.portfolio_value = np.random.uniform( 0.8, 1.2 )
        self.weights = np.random.dirichlet( np.ones( (1 + self.n_risky_assets,) ) )
                         
        # Set the asset process
        process_args = dict(n_risky_assets=self.n_risky_assets, \
                            trading_periods_per_year=self.trading_periods_per_year)
        if self.asset_process_name == 'lognormal-static':
            self.asset_process = distributions.LognormalStaticProcess(**process_args)                        
        elif self.asset_process_name == 'normal-static':
            self.asset_process = distributions.NormalStaticProcess(**process_args)
        else:
            raise ValueError('Unsupported process name: {}'.format(self.asset_process_name) )
        asset_params = self.asset_process.params_to_array( self.asset_process.get_parameters() )
        return np.hstack( [ [self.portfolio_value], self.weights[1:].ravel(), asset_params.ravel() ] )
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    
    def parse_observation( self, obs ):
        N = self.n_risky_assets
        wts = np.hstack( [ 1 - sum(obs[1:(1+N)]), obs[1:(1+N)].ravel() ] )
        process_params = self.asset_process.array_to_params(obs[N+1:])
        return ObsParams( portfolio_value=obs[0], weight=wts, process_params=process_params )
            
    def _get_observation( self, new_ptf_val, new_weights ):
        asset_params = self.asset_process.params_to_array( self.asset_process.get_parameters() )
        return np.hstack( [ [ new_ptf_val], new_weights[1:].ravel(), asset_params.ravel() ] )
        
    def _get_random_returns(self):
        """Generate arithmetic (simple) asset class returns based on the asset process.""" 
        return self.asset_process.distrib.sample()
        
    def _calculate_reward( self, action, old_ptf_val, new_ptf_val ):
        """Calculate the reward for a new time step, as the change in discounted utility."""
        gamma = self.get_gamma()
        reward = self.utility_fun( new_ptf_val ) - 1/gamma * self.utility_fun( old_ptf_val )            
        return reward
    
    def get_gamma(self):
        """Get the discount factor from the risk-free rate."""
        return np.exp( -self.asset_process.risk_free_rate / self.trading_periods_per_year )        
    