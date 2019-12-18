import gym
import numpy as np
from collections import namedtuple
import distributions

# Define a named tuple to help with parameters
fields = [ 'timestamp', 'ptf_val', 'ptf_wts', 'bmk_val', 'bmk_wts', 'process_params' ]
defaults = (None,) * len(fields)
StateVariables = namedtuple( 'StateVariables', fields, defaults=defaults )

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
                  min_short=0.0,
                  max_long=1.0,
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
        assert not np.any(np.isnan(action)), 'Actions must always be valid numbers.'

        # Generate random asset returns from the asset process
        asset_rtns = self.asset_process.generate_random_returns()

        # Constrain the action to lie within the allowable weight space
        chg_wts = self._project_action_to_allowable_weight_space(action)
        
        # Get the new portfolio value, treating the action as the change in weights of risky assets
        new_ptf_val, new_ptf_wts = self._calculate_portfolio_value( ptf_val=self.state_vars.ptf_val, \
                            old_wts=self.state_vars.ptf_wts, asset_rtns=asset_rtns, chg_wts=chg_wts )
        new_bmk_val, new_bmk_wts = self._calculate_portfolio_value( ptf_val=self.state_vars.bmk_val, \
                            old_wts=self.state_vars.bmk_wts, asset_rtns=asset_rtns )
        
        # Save the state process, and Evolve the asset process (if it is non-static)
        old_state_vars = self.state_vars
        self.asset_process.evolve()
        self.state_vars = StateVariables( timestamp=1 + old_state_vars.timestamp,
                                          ptf_val=new_ptf_val, ptf_wts=new_ptf_wts,
                                          bmk_val=new_bmk_val, bmk_wts=new_bmk_wts, 
                                          process_params=self.asset_process.get_parameters() )
        obs = self._get_observation_from_state_vars()
        
        # Calculate the reward from the old and new state parameters        
        reward = self._calculate_reward( old_state_vars, self.state_vars )
        done = (self.state_vars.ptf_val == 0) or \
               (self.state_vars.timestamp == self.n_years * self.n_periods_per_year) 
        info = {'asset_rtns' : asset_rtns, 'chg_wts' : chg_wts }        
        return obs, reward, done, info
        
    def reset(self):
        """Reset the state of the environment to an initial state."""        
        ptf_val, ptf_wts, bmk_val, bmk_wts = self._generate_initial_portfolios()
        process_params = self._generate_initial_process()
        self.state_vars = StateVariables( timestamp=0,
                                          ptf_val=ptf_val, ptf_wts=ptf_wts,
                                          bmk_val=bmk_val, bmk_wts=bmk_wts, 
                                          process_params=process_params )
        return self._get_observation_from_state_vars()
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    
    def parse_observation( self, obs ):
        N = self.n_risky_assets
        
        timestamp=obs[0]

        ptf_val=obs[1]
        ptf_wts_risky = obs[2:2+N].ravel()
        ptf_wt_cash = 1 - sum(ptf_wts_risky)        
        ptf_wts = np.hstack( [ ptf_wt_cash, ptf_wts_risky ] )

        if self.benchmark_weights is not None:
            bmk_val=obs[2+N]
            bmk_wts_risky = obs[3+N:3+2*N].ravel()
            bmk_wt_cash = 1 - sum(ptf_wts_risky)        
            bmk_wts = np.hstack( [ bmk_wt_cash, bmk_wts_risky ] )        
        
            process_params = self.asset_process.array_to_params(obs[3+2*N:])
        else:
            bmk_val = bmk_wts = None
            process_params = self.asset_process.array_to_params(obs[2+N:])
            
        return StateVariables( timestamp=timestamp,
                               ptf_val=ptf_val, ptf_wts=ptf_wts, 
                               bmk_val=bmk_val, bmk_wts=bmk_wts, 
                               process_params=process_params )
    
    def set_utility_function(self, utility_fun):
        if utility_fun is None:
            self.utility_fun = distributions.ExponentialUtilityFunction
        else:
            self.utility_fun = utility_fun
            
    def _get_observation_from_state_vars(self):
        sv = self.state_vars
        if self.benchmark_weights is not None:
            return np.hstack( [ sv.timestamp, 
                                sv.ptf_val, sv.ptf_wts[1:].ravel(), 
                                sv.bmk_val, sv.bmk_wts[1:].ravel(), 
                                self.asset_process.params_to_array(sv.process_params) ] )
        else:
            return np.hstack( [ sv.timestamp, 
                                sv.ptf_val, sv.ptf_wts[1:].ravel(), 
                                self.asset_process.params_to_array(sv.process_params) ] )        
        
    def _get_asset_process_parans(self):
        return self.asset_process.params_to_array( self.asset_process.get_parameters() )
        
    def _get_random_returns(self):
        """Generate arithmetic (simple) asset class returns based on the asset process.""" 
        return self.asset_process.distrib.random()
        
    def _calculate_reward( self, old_state_vars, new_state_vars ):
        """Calculate the reward for a new time step, as the change in discounted utility."""
        if 'total-wealth' == self.objective:
            x_old = old_state_vars.ptf_val
            x_new = new_state_vars.ptf_val
        elif 'relative-profit' == self.objective:
            x_old = old_state_vars.ptf_val - old_state_vars.bmk_val
            x_new = new_state_vars.ptf_val - new_state_vars.bmk_val
        else:
            raise ValueError( 'Unsupported objective: {}'.format(objective) )

        # Discount the objective before applying the utility function
        gamma = self.get_gamma()            
        reward = self.utility_fun( gamma * x_new ) - self.utility_fun( x_old )
        return reward
    
    def get_gamma(self):
        """Get the discount factor from the risk-free rate."""
        return np.exp( -self.asset_process.risk_free_rate / self.n_periods_per_year )        
        
    def _setup_spaces(self):
        """Set up the action and observation spaces."""
        # Actions are the changes in weights of all assets (cash + the risky assets)
        N = self.n_risky_assets
        wt_interval = self.max_long - self.min_short
        self.action_space = gym.spaces.Box( low  = -wt_interval * np.ones( (N+1,) ), 
                                            high = +wt_interval * np.ones( (N+1,) ) )
        # Define the dimensions of the observation space, starting with the portfolio value & weights
        param_ranges = self.asset_process.get_parameter_ranges()
        min_ptf_val, max_ptf_val = 0, np.inf
        low  = np.hstack( [ min_ptf_val, -wt_interval * np.ones((N,)) ] )
        high = np.hstack( [ max_ptf_val, +wt_interval * np.ones((N,)) ] )
                
        if self.benchmark_weights is not None:
            # Repeat the low / high limits for the benchmark
            low = np.hstack( [ low, low ] )
            high = np.hstack( [ high, high ] )
            
        # Add the parameter ranges and the timestamp
        low  = np.hstack( [ 0, low, param_ranges.low ] )
        high = np.hstack( [ self.n_years * self.n_periods_per_year, high, param_ranges.high ] )        
        self.observation_space = gym.spaces.Box( low=low, high=high )
    
    def _calculate_portfolio_value( self, ptf_val, old_wts, asset_rtns, chg_wts=None ):
        """Calculate the new portfolio value and new weights.
           This method subtracts any transaction costs from the cash allocation, and then
             rescales the weights so that they sum to 1."""
        
        # When benchmark is None, there are no weights assigned. In this case, we return None
        if ptf_val is None or old_wts is None:
            return None, None
        
        if chg_wts is None:
            chg_wts = np.zeros_like(old_wts)
        
        if self.trans_costs > 0:
            # Fund transaction costs from the cash allocation
            asset_mkt_val = ptf_val * old_wts
            asset_mkt_val[0] -= self.trans_costs * np.sum( np.abs(chg_wts[1:]) ) * ptf_val
            ptf_val = asset_mkt_val.sum()
            old_wts = asset_mkt_val / ptf_val
            
        new_wts = old_wts + chg_wts            
        assert np.isclose( new_wts.sum(), 1.0 ), 'Weights must sum to 1.'
        ptf_rtn = np.matmul( new_wts, asset_rtns.T )[0]
        new_ptf_val = max( 0, ptf_val * ( 1 + ptf_rtn ) )
        return new_ptf_val, new_wts
               
    def _generate_initial_portfolios(self):
        ptf_val = 1.0 - 0.1 * self.np_random.randn()
        ptf_wts = self.np_random.dirichlet( np.ones( (1 + self.n_risky_assets,) ) )
        bmk_val = ptf_val
        bmk_wts = self.benchmark_weights
        return ptf_val, ptf_wts, bmk_val, bmk_wts 
            
    def _generate_initial_process(self):
        process_args = dict(n_risky_assets=self.n_risky_assets, np_random=self.np_random, \
                            n_periods_per_year=self.n_periods_per_year)
        if self.asset_process_name == 'lognormal-static':
            self.asset_process = distributions.LognormalStaticProcess(**process_args)                        
        elif self.asset_process_name == 'normal-static':
            self.asset_process = distributions.NormalStaticProcess(**process_args)
        else:
            raise ValueError('Unsupported process name: {}'.format(self.asset_process_name) )
        return self.asset_process.get_parameters()    
    
    def _project_action_to_allowable_weight_space(self, action):

        # Get the direction of the change in weights
        wt_chg_direction = np.hstack( [ -action.sum(), action ] )
        old_wts = self.state_vars.ptf_wts
        new_wts = old_wts + wt_chg_direction
        
        # Find maximally violated constraint
        constraint_vals = np.maximum( self.min_short - new_wts, new_wts - self.max_long )
        if np.all( constraint_vals < 1e-5 ):
            wt_chg = wt_chg_direction
        else:
            idx = np.argmax(constraint_vals)
            vec_len = wt_chg_direction[idx]
            wt = old_wts[idx]
            dist_to_bdry = wt - self.min_short if vec_len < 0 else self.max_long - wt
            wt_chg = dist_to_bdry / np.abs(vec_len) * wt_chg_direction
            
        assert np.isclose( wt_chg.sum(), 0 ), 'Changes in weights must sum to 0.'
        return wt_chg
        
        