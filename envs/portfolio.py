import gym
import numpy as np
from collections import namedtuple
import envs.portfolio_distributions

# Define a named tuple to help with parameters
fields = [ 'timestamp', 'ptf_asset_vals', 'bmk_asset_vals', 'process_params' ]
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
                  is_recurrent=True,                 
                  asset_process=None, 
                  max_episode_steps=200
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
        
        assert self.asset_process is None or isinstance( self.asset_process, \
                    envs.portfolio_distributions.AssetProcess ), 'Invalid input for asset_process.'
                
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
        ptf_kwargs = dict( ptf_asset_vals=self.state_vars.ptf_asset_vals, asset_rtns=asset_rtns, \
                                                                                  chg_wts=chg_wts)
        new_ptf_asset_vals = self._calc_ptf_asset_vals(**ptf_kwargs)

        if self.benchmark_weights is None:
            new_bmk_asset_vals = None
        else:
            bmk_kwargs = dict( ptf_asset_vals=self.state_vars.bmk_asset_vals, asset_rtns=asset_rtns )            
            new_bmk_asset_vals = self._calc_ptf_asset_vals(**bmk_kwargs)
                    
        # Save the state process, and Evolve the asset process (if it is non-static)
        old_state_vars = self.state_vars
        self.asset_process.evolve()
        new_state_vars = StateVariables( timestamp=1 + old_state_vars.timestamp,
                                         ptf_asset_vals=new_ptf_asset_vals,
                                         bmk_asset_vals=new_bmk_asset_vals,
                                         process_params=self.asset_process.get_parameters() )
        
        # Calculate the reward from the old and new state parameters        
        reward = self._calculate_reward( old_state_vars, new_state_vars )
        done = new_ptf_asset_vals.sum() <= 1e-4 or \
               self.state_vars.timestamp >= self.max_episode_steps 
        info = {'asset_rtns' : asset_rtns, 'chg_wts' : chg_wts }
        
        # Rescale portfolio values so they sum to 1 if the environment is recurrent
        if self.is_recurrent:
            new_ptf_asset_vals /= new_ptf_asset_vals.sum()
            if self.benchmark_weights is not None:
                new_bmk_asset_vals /= new_bmk_asset_vals.sum()
            self.state_vars = StateVariables( timestamp=1 + old_state_vars.timestamp,
                                              ptf_asset_vals=new_ptf_asset_vals,
                                              bmk_asset_vals=new_bmk_asset_vals,
                                              process_params=self.asset_process.get_parameters() )
        else:
            self.state_vars = new_state_vars
        obs = self._get_observation_from_state_vars()        
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
        if not self.is_recurrent:
            timestamp = obs[0]
            obs = obs[1:]
        else:
            timestamp = None
            
        ptf_asset_vals=obs[2:N+3].ravel()
        if self.benchmark_weights is not None:
            bmk_asset_vals=obs[3+N:4+2*N].ravel()
            process_params = self.asset_process.array_to_params(obs[4+2*N:])
        else:
            bmk_asset_vals = None
            process_params = self.asset_process.array_to_params(obs[3+N:])
            
        return StateVariables( timestamp=timestamp,
                               ptf_asset_vals=ptf_asset_vals, 
                               bmk_asset_vals=bmk_asset_vals,  
                               process_params=process_params )
    
    def set_utility_function(self, utility_fun):
        if utility_fun is None:
            self.utility_fun = envs.portfolio_distributions.ExponentialUtilityFunction(alpha=2)
        else:
            self.utility_fun = utility_fun
            
    def set_asset_process( self, new_asset_process ):
        self.asset_process = new_asset_process        
        sv = self.state_vars
        self.state_vars = StateVariables( timestamp=sv.timestamp, 
                                          ptf_asset_vals=sv.ptf_asset_vals,
                                          bmk_asset_vals=sv.bmk_asset_vals, 
                                          process_params=new_asset_process.get_parameters() )
            
    def _get_observation_from_state_vars(self):
        sv = self.state_vars
        obs = []
        if not self.is_recurrent:
            obs.append( sv.timestamp ) 
            
        obs.append( sv.ptf_asset_vals.ravel() )
        if self.benchmark_weights is not None:
            sv.bmk_asset_vals.ravel()
            
        obs.append( self.asset_process.params_to_array(sv.process_params) )
        return np.hstack(obs)
        
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
        if self.is_recurrent:
            return 1.0
        else:
            return 1 / np.power( 1 + self.asset_process.risk_free_rate, 1 / self.n_periods_per_year )
        
    def _setup_spaces(self):
        """Set up the action and observation spaces."""
        # Actions are the changes in weights of risky
        N = self.n_risky_assets
        self.action_space = gym.spaces.Box( low  = -np.ones( (N,) ), 
                                            high = +np.ones( (N,) ) )
        
        # Define the dimensions of the observation space, starting with the portfolio value & weights
        param_ranges = self.asset_process.get_parameter_ranges()
        min_asset_val, max_asset_val = -np.inf, np.inf
        low  = min_asset_val * np.ones((N+1,))
        high = max_asset_val * np.ones((N+1,))
                
        if self.benchmark_weights is not None:
            # Repeat the low / high limits for the benchmark
            low = np.hstack( [ low, low ] )
            high = np.hstack( [ high, high ] )
            
        # Add the parameter ranges
        low  = np.hstack( [ low, param_ranges.low ] )
        high = np.hstack( [ high, param_ranges.high ] )
        
        # Add the timestamp, for non-recurrent environments
        if not self.is_recurrent:
            low  = np.hstack( [ 0, low ] )
            high = np.hstack( [ self.max_episode_steps, high ] )
            
        self.observation_space = gym.spaces.Box( low=low, high=high )
    
    def _calc_ptf_asset_vals( self, ptf_asset_vals, asset_rtns, chg_wts=None ):
        """Calculate the new asset values.
           This method subtracts any transaction costs from the cash allocation."""        
        if chg_wts is None:
            chg_wts = np.zeros_like(asset_rtns)
        else:
            assert np.isclose( chg_wts.sum(), 0 ), 'Changes in weights must sum to 0.'
        
        # Obtain the new weights from the old, and deduct transaction costs from cash
        tot_ptf_val = ptf_asset_vals.sum()
        ptf_asset_vals_ex_cost = ptf_asset_vals + chg_wts * tot_ptf_val
        ptf_asset_vals_ex_cost[0] -= np.sum( self.trans_costs * np.abs(chg_wts[1:]) ) * tot_ptf_val
        return ptf_asset_vals_ex_cost.ravel() * (1 + asset_rtns.ravel() )
               
    def _generate_initial_portfolios(self):
        if self.is_recurrent:
            initial_ptf_val = 1.0
        else:
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
        self.asset_process = envs.portfolio_distributions.LognormalStaticProcess(**process_args)       

        return self.asset_process.get_parameters()    
    
    def _project_action_to_allowable_weight_space(self, action):
        # Get the direction of the change in weights (the action is the change in risky asset weights)
        risky_weights = action
        wt_chg_direction = np.hstack( [ -risky_weights.sum(), risky_weights ] )
        old_wts = self.state_vars.ptf_asset_vals / self.state_vars.ptf_asset_vals.sum()
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
            
        assert ~np.any( np.isnan(wt_chg) ), 'Changes in weights cannot be NaN.'
        assert np.isclose( wt_chg.sum(), 0 ), 'Changes in weights must sum to 0.'
        return wt_chg
        
        