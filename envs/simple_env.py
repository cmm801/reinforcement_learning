import gym
import numpy as np
from collections import namedtuple
import scipy.optimize
from scipy.optimize import fminbound

# Define a named tuple to help with parameters
fields = [ 'position' ]
defaults = (None,) * len(fields)
StateVariables = namedtuple( 'StateVariables', fields, defaults=defaults )

class SimpleEnv(gym.Env):

    def __init__(self, reward_func, low, high, max_steps=300, tol=1e-4, action_step_size=0.10 ):
        self.reward_func = reward_func
        self._max_episode_steps = max_steps
        self.tol = tol
        self._step_counter = 0
        self.n_dims = np.array(low).size
        self.action_space = gym.spaces.Box(low=-action_step_size * np.ones((self.n_dims,)), 
                                           high=+action_step_size * np.ones((self.n_dims,)) ) 
        self.observation_space = gym.spaces.Box(low=np.array(low), high=np.array(high) )
        
        # Find the maximum reward and its location
        self.x_max = self._get_reward_func_maximum()
        self.reward_max = self.reward_func(self.x_max)
        
    def reset(self):
        self._step_counter = 0        
        obs = np.array( self.observation_space.sample() ).ravel()
        self.state_vars = StateVariables( position=obs )
        return obs
    
    def step(self, action):
        assert self.action_space.contains(action), 'Action lies outside of allowed values.'        
        reward, obs = self.calculate_reward(action )
        self.state_vars = StateVariables( position=obs )
        self._step_counter += 1
        done = self._step_counter == self._max_episode_steps or \
                        np.all( np.isclose( self.x_max, obs, atol=self.tol) )
        info = {}
        return obs, reward, done, info
        
    def seed(self, seed=None):
        #self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def calculate_reward(self, action ):
        old_pos = self.state_vars.position
        old_val = self.reward_func( old_pos )
        new_pos = np.clip( self.state_vars.position + action,
                      self.observation_space.low, self.observation_space.high ).ravel()
        new_val = self.reward_func(new_pos)
        reward = new_val - old_val
        return reward, new_pos
        
    def set_state(self, state):
        self.state_vars = StateVariables( position=state )
        
    def _get_reward_func_maximum(self):
        x0 = np.zeros_like( self.observation_space.low )
        bounds = scipy.optimize.Bounds( self.observation_space.low, self.observation_space.high )
        cost_func = lambda x : -self.reward_func(x)
        res = scipy.optimize.minimize( cost_func, x0, bounds=bounds, method='SLSQP' )
        if res.success:
            return res.x
        else:
            raise RuntimeError( 'Optimization does not converge.' )    