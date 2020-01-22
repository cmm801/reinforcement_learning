import gym
import numpy as np
from collections import namedtuple
import envs.portfolio_distributions

# Define named tuples to help manage parameters
StateVariables = namedtuple( 'StateVariables', 
                    field_names=[ 'timestamp', 'portfolio', 'benchmark', 'process_params' ], 
                    defaults=[None] * 4 )
PortfolioVariables = namedtuple( 'PortfolioVariables', 
                    field_names=[ 'value', 'weights', 'gamma_params' ], 
                    defaults=[None, None, None ] )

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
                  objective='total-wealth',
                  benchmark_weights=None,
                  utility_fun=None, 
                  trans_costs=0.0,
                  n_periods_per_year=12,
                  n_years=10,
                  is_static=True,
                  asset_process_name='lognormal-static'
                ):
        """This environment takes the parameters of its distribution as inputs."""
        super( self.__class__, self).__init__()

        # Save input arguments        
        input_args = locals()
        for field_name, input_val in input_args.items():
            self.__setattr__( field_name, input_val )
        self.set_utility_function(utility_fun)
        if benchmark_weights is not None:
            self.benchmark_weights = np.array( benchmark_weights )
        
        # Set the internal random number generator, whose seed can be used to obtain reproducible results
        self.np_random = np.random.RandomState()
        
        # Initialize the asset process and state variables
        self.reset()
        
        # Set up action and obseration spaces
        self._setup_spaces()
                         
    def seed(self, seed=None):
        #self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def step(self, action):                
        # Generate random asset returns from the asset process
        asset_rtns = self.asset_process.generate_random_returns()
        
        # Get the new portfolio value, treating the action as the change in weights of risky assets
        ptf_kwargs = dict( ptf_asset_vals=self.state_vars.ptf_asset_vals, asset_rtns=asset_rtns, \
                                                                                  new_wts=action)
        new_ptf_asset_vals = self._calc_ptf_asset_vals(**ptf_kwargs)

        if self.benchmark_weights is None:
            new_bmk_asset_vals = None
        else:
            bmk_kwargs = dict( ptf_asset_vals=self.state_vars.bmk_asset_vals, asset_rtns=asset_rtns )            
            new_bmk_asset_vals = self._calc_ptf_asset_vals(**bmk_kwargs)
        
        # Save the state process, and Evolve the asset process (if it is non-static)
        old_state_vars = self.state_vars
        self.asset_process.evolve()
        self.state_vars = StateVariables( timestamp=1 + old_state_vars.timestamp,
                                          ptf_asset_vals=new_ptf_asset_vals, 
                                          bmk_asset_vals=new_bmk_asset_vals,
                                          process_params=self.asset_process.get_parameters() )
        obs = self._get_observation_from_state_vars()
        
        # Calculate the reward from the old and new state parameters        
        reward = self._calculate_reward( old_state_vars, self.state_vars )
        done = np.isclose( self.state_vars.ptf_asset_vals.sum(), 0.0 ) or \
               (self.state_vars.timestamp == self.n_years * self.n_periods_per_year) 
        info = {'asset_rtns' : asset_rtns }        
        return obs, reward, done, info
        
    def reset(self):
        """Reset the state of the environment to an initial state."""        
        ptf_asset_vals, bmk_asset_vals = self._generate_initial_portfolios()
        process_params = self._generate_initial_process()
        self.state_vars = StateVariables( timestamp=0,
                                          ptf_asset_vals=ptf_asset_vals, 
                                          bmk_asset_vals=bmk_asset_vals,  
                                          process_params=process_params )
        return self._get_observation_from_state_vars()
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    
    def parse_observation( self, obs ):
        N = self.n_risky_assets        
        timestamp=obs[0]
        ptf_asset_vals=obs[1:N+2].ravel()
        if self.benchmark_weights is not None:
            bmk_asset_vals=obs[2+N:3+2*N].ravel()        
            process_params = self.asset_process.array_to_params(obs[3+2*N:])
        else:
            bmk_asset_vals = None
            process_params = self.asset_process.array_to_params(obs[2+N:])
            
        return StateVariables( timestamp=timestamp,
                               portfolio=portfolio, 
                               benchmark=benchmark,  
                               process_params=process_params )
    
    def set_utility_function(self, utility_fun):
        if utility_fun is None:
            self.utility_fun = envs.portfolio_distributions.ExponentialUtilityFunction
        else:
            self.utility_fun = utility_fun
            
    def set_asset_process( self, new_asset_process ):
        self.asset_process = new_asset_process        
        sv = self.state_vars
        self.state_vars = StateVariables( timestamp=sv.timestamp,
                                          portfolio=sv.portfolio,
                                          benchmark=sv.benchmark, 
                                          process_params=new_asset_process.get_parameters() )
        return self._get_observation_from_state_vars()
            
    def _get_observation_from_state_vars(self):
        sv = self.state_vars
        if self.benchmark_weights is not None:
            return np.hstack( [ sv.timestamp, 
                                sv.portfolio.value,
                                sv.portfolio.weights.ravel(),
                                sv.benchmark.value,
                                sv.benchmark.weights.ravel(),
                                self.asset_process.params_to_array(sv.process_params) ] )
        else:
            return np.hstack( [ sv.timestamp, 
                                sv.portfolio.value,
                                sv.portfolio.weights.ravel(),
                                self.asset_process.params_to_array(sv.process_params) ] )        
        
    def _get_asset_process_parans(self):
        return self.asset_process.params_to_array( self.asset_process.get_parameters() )
        
    def _get_random_returns(self):
        """Generate arithmetic (simple) asset class returns based on the asset process.""" 
        return self.asset_process.distrib.random()
        
    def _calculate_reward( self, old_state_vars, new_state_vars ):
        """Calculate the reward for a new time step, as the change in discounted utility."""
        if 'total-wealth' == self.objective:
            x_old = old_state_vars.ptf_asset_vals.sum()
            x_new = new_state_vars.ptf_asset_vals.sum()
        elif 'relative-profit' == self.objective:
            x_old = old_state_vars.ptf_asset_vals.sum() - old_state_vars.bmk_asset_vals.sum()
            x_new = new_state_vars.ptf_asset_vals.sum() - new_state_vars.bmk_asset_vals.sum()
        else:
            raise ValueError( 'Unsupported objective: {}'.format(objective) )

        # Discount the objective before applying the utility function
        gamma = self.get_gamma()            
        reward = self.utility_fun( gamma * x_new ) - self.utility_fun( x_old )
        return reward
    
    def get_gamma(self):
        """Get the discount factor from the risk-free rate."""
        return - 1 + np.power( 1 + self.asset_process.risk_free_rate, 1 / self.n_periods_per_year )
        
    def _setup_spaces(self):
        """Set up the action and observation spaces."""
        # Actions are the new weights of all assets (cash + the risky assets)
        N = self.n_risky_assets
        self.action_space = gym.spaces.Box( low=np.zeros( (N+1,) ), 
                                            high=np.ones( (N+1,) ) )
        
        # The parameters describing a portfolio are the total value and then the weights of each asset
        low  = np.hstack( [ 0, np.zeros((N+1,)) ] )
        high = np.hstack( [ np.inf, np.ones((N+1,)) ] )
                
        if self.benchmark_weights is not None:
            # Repeat the low / high limits for the benchmark if it is used
            low = np.hstack( [ low, low ] )
            high = np.hstack( [ high, high ] )
            
        # Add the parameter ranges and the timestamp
        param_ranges = self.asset_process.get_parameter_ranges()        
        low  = np.hstack( [ 0, low, param_ranges.low ] )
        high = np.hstack( [ self.n_years * self.n_periods_per_year, high, param_ranges.high ] )        
        self.observation_space = gym.spaces.Box( low=low, high=high )
    
    def _calc_ptf_asset_vals( self, ptf_asset_vals, asset_rtns, new_wts=None ):
        """Calculate the new asset values.
           This method subtracts any transaction costs from the cash allocation."""        
        if new_wts is None:
            new_ptf_asset_vals = ptf_asset_vals
        else:
            new_ptf_asset_vals = new_wts * ptf_asset_vals.sum()
        
        # Obtain the new weights from the old, and deduct transaction costs from cash
        chg_ptf_vals = new_ptf_asset_vals - ptf_asset_vals
        ptf_asset_vals_ex_cost = new_ptf_asset_vals
        ptf_asset_vals_ex_cost[0] -= np.sum( self.trans_costs * np.abs(chg_ptf_vals[1:]) )
        return ptf_asset_vals_ex_cost.ravel() * (1 + asset_rtns.ravel() )
               
    def _generate_initial_portfolios(self):
        initial_ptf_val = 1.0 - 0.1 * self.np_random.randn()
        initial_ptf_wts = self.np_random.dirichlet( np.ones( (1 + self.n_risky_assets,) ) )
        ptf_asset_vals = initial_ptf_val * initial_ptf_wts

        if self.benchmark_weights is not None:
            bmk_asset_vals = initial_ptf_val * self.benchmark_weights 
        else:
            bmk_asset_vals = None            
        return ptf_asset_vals, bmk_asset_vals
            
    def _generate_initial_process(self):
        process_args = dict(n_risky_assets=self.n_risky_assets, np_random=self.np_random, \
                            n_periods_per_year=self.n_periods_per_year)
        if self.asset_process_name == 'lognormal-static':
            self.asset_process = envs.portfolio_distributions.LognormalStaticProcess(**process_args)                        
        elif self.asset_process_name == 'normal-static':
            self.asset_process = envs.portfolio_distributions.NormalStaticProcess(**process_args)
        else:
            raise ValueError('Unsupported process name: {}'.format(self.asset_process_name) )
        return self.asset_process.get_parameters()    
