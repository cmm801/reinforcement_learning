import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from IPython.display import clear_output
from tqdm import trange

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class ActorCritic():
    def __init__(self,
                 env,
                 entropy_coef=0.01,
                 reward_scale=1.,
                 actor_model=None, 
                 critic_model=None,
                 actor_learning_rate=0.001,
                 critic_learning_rate=0.001,
                 action_space_distr='normal', 
                 scale_model_inputs=True
                ):

        # Set parameters
        if isinstance( env, EnvBatch ):
            self.env = env
        else:
            env_name = env.unwrapped.spec.id
            self.env = EnvBatch(env_name)
        
        # Extract action space / observation space information from the environment
        self._env_helper = EnvHelper(env, action_space_distr=action_space_distr)
        
        # Get the standard scaler, if user wants to scale the inputs
        self.scale_model_inputs = scale_model_inputs
        self._init_standard_scaler(env, scale_model_inputs=scale_model_inputs)
        
        # Set parameters
        self.entropy_coef = entropy_coef    # The coefficient of entropy used in the loss function
        self.reward_scale = reward_scale    # rescale rewards to keep gradients from exploding
        
        # Set optimizers
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=actor_learning_rate)        
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=critic_learning_rate)
        
        # Set Actor model
        if actor_model is None:
            self.actor_model = get_mlp_actor( env, action_space_distr=action_space_distr )
        else:
            self.actor_model = actor_model

        # Set Critic models
        if critic_model is None:
            self.critic_model = get_mlp_critic( env )
        else:
            self.critic_model = critic_model                        
    
    def choose_action_from_state(self, state ):
        """pick actions given numeric agent outputs (np arrays)"""
        distrib_params = self.actor_model(state)
        return self._env_helper.sample_actions(distrib_params)
        
    def get_state_values(self, state):
        scaled_state = self._apply_scaling(state)
        state_value_raw = self.critic_model(scaled_state)
        return tf.reshape( state_value_raw, shape=(-1,) )        
    
    def train_step(self, states, next_states, actions, rewards, is_done, gamma=0.99):
        """Train the neural network from the states, rewards and transitions."""
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.actor_model.trainable_variables)
            tape.watch(self.critic_model.trainable_variables)
            
            # Get the probability distribution over possible actions, conditional on the state
            distrib = self._get_action_distributions(states)

            # Get the estimated value of the current and next state
            state_values = self.get_state_values(states)       
            next_state_values = self.get_state_values(next_states)        
            next_state_values = next_state_values * (1 - is_done)

            # log-probabilities only for agent's chosen actions
            logp_actions = distrib.log_prob(actions)

            # Actor loss calculation (policy gradient approach)
            advantage = rewards + gamma * next_state_values - state_values
            entropy = distrib.entropy()
            actor_loss = -tf.reduce_mean( logp_actions * tf.stop_gradient(advantage) ) \
                         -self.entropy_coef * tf.reduce_mean(entropy)

            # Critic loss calculation
            target_state_values = rewards + gamma * next_state_values
            critic_loss = tf.reduce_mean( tf.square( state_values - tf.stop_gradient(target_state_values) ) )
            
            
        grads_actor = tape.gradient( actor_loss, self.actor_model.trainable_variables )
        grads_critic = tape.gradient( critic_loss, self.critic_model.trainable_variables )
        del tape

        # Apply gradients      
        self.actor_optimizer.apply_gradients( zip( grads_actor, self.actor_model.trainable_variables ) ) 
        self.critic_optimizer.apply_gradients( zip( grads_critic, self.critic_model.trainable_variables ) )        
        
        return actor_loss, critic_loss, entropy

    def _get_action_distributions(self, state):
        scaled_state = self._apply_scaling(state)
        params = self.actor_model(scaled_state)        
        return self._env_helper.get_action_distributions(params)  
    
    def _apply_scaling(self, state):
        if self.scale_model_inputs:
            return self._standard_scaler.transform(state)
        else:
            return state
            
    def play_games(self, n_games=1):
        """Plays an a game from start till done, returns per-game rewards """

        # Create a temporary environment to play some games
        tmp_env = make_env( self.env.environ_name, max_steps=self.env.max_steps)

        game_rewards = []
        for _ in range(n_games):
            state = np.array([tmp_env.reset()])

            total_reward = 0
            while True:
                action = self.choose_action_from_state(state)
                tstate, reward, done, info = tmp_env.step(action)
                total_reward += reward
                if done: break

            # We rescale the reward back to ensure compatibility
            # with other evaluations.
            game_rewards.append(total_reward / self.reward_scale)
        return game_rewards       
    
    def run(self, n_iters=2000, gamma=0.99, n_games_per_plot_update=3, n_iters_per_plot_update=100 ):
        """Run the training for a number of iterations, and regularly plot the progress."""

        # Get the initial states
        batch_states = self.env.reset()
        rewards_mean, rewards_max, rewards_min = [], [], []
        entropy_history = []        

        for i in trange(n_iters):
            batch_actions = self.choose_action_from_state(batch_states)
            batch_next_states, batch_rewards, batch_done, _ = self.env.step(batch_actions)

            # Train the neural network from the states, rewards and transitions
            actor_loss, critic_loss, ent_t = self.train_step( batch_states, batch_next_states, \
                                          batch_actions, batch_rewards, batch_done, gamma=gamma )
            batch_states = batch_next_states
            entropy_history.append(np.mean(ent_t))

            if i % 100 == 0:
                scores = self.play_games(n_games=n_games_per_plot_update)
                rewards_min.append(np.min(scores))                    
                rewards_mean.append(np.mean(scores))
                rewards_max.append(np.max(scores))                    

                clear_output(True)
                plt.figure(figsize=[14, 6])
                plt.subplot(1, 2, 1)
                plt.plot(rewards_mean, label='avg_rewards')
                plt.plot(ewma(np.array(rewards_mean), span=10), marker='.', label='avg rewards ewma@10')
                plt.plot(ewma(np.array(rewards_min), span=10), marker='.', label='min rewards ewma@10')
                plt.plot(ewma(np.array(rewards_max), span=10), marker='.', label='max rewards ewma@10')
                plt.title("Session rewards")
                plt.grid()
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(entropy_history, label='entropy')
                plt.plot(ewma(np.array(entropy_history), span=1000), label='entropy ewma@1000')
                plt.title("Policy entropy")
                plt.grid()
                plt.legend()
                plt.show()    
        return rewards_mean, rewards_min, rewards_max, entropy_history

    def _init_standard_scaler(self, env, scale_model_inputs=True):
        if scale_model_inputs:
            samples = np.array(
                [env.observation_space.sample() for x in range(10000)])
            self._standard_scaler = StandardScaler()
            self._standard_scaler.fit(samples)
        else:
            self._standard_scaler = None
    
class EnvBatch():
    """A class that allows us to play several games simultaneously, which improves the 
            convergence properties of A2C."""
    
    def __init__(self, environ_name, n_envs = 1, max_steps=None):
        """ Create n_envs different environments for the possibility of asynchronous training. """
        self.environ_name = environ_name
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.envs = [make_env(environ_name, max_steps=max_steps) for _ in range(n_envs)]        
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space        
        
    def reset(self):
        """ Reset all games and return [n_envs, *obs_shape] observations """
        return np.vstack( [ env.reset().ravel().T for env in self.envs] )
    
    def step(self, actions):
        """
        Send a vector[batch_size] of actions into respective environments
        :returns: observations[n_envs, *obs_shape], rewards[n_envs], done[n_envs,], info[n_envs]
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, done, info = map(np.array, zip(*results))
        new_obs = np.vstack( [ x.ravel().T for x in new_obs ] )
        
        # reset environments automatically
        for i in range(len(self.envs)):
            if done[i]:
                new_obs[i] = self.envs[i].reset()
        
        return new_obs, rewards, done, info   
    

class EnvHelper():
    
    def __init__(self, env, action_space_distr='normal'):    
        if isinstance(env, str):
            env = make_env(env)   
        elif isinstance(env, EnvBatch):
            env = env.envs[0]
            
        self.action_space_distr = action_space_distr
        self._set_action_space(env.action_space, distrib=self.action_space_distr)
        self._set_observation_space(env.observation_space)

    def sample_actions(self, distrib_params):
        return self.action_helper.sample(distrib_params)
              
    def get_action_distributions(self, distrib_params):
        return self.action_helper.get_distributions(distrib_params)
                
    def _set_action_space(self, action_space, distrib=None):
        self.action_helper = GymSpaceHelper(action_space, distrib=distrib)
        self.action_dim = self.action_helper.dim
        self.action_type = self.action_helper.type
        self.n_action_vars = self.action_helper.n_vars
             
    def _set_observation_space(self, obs_space):
        self.obs_helper = GymSpaceHelper(obs_space)
        self.obs_dim = self.obs_helper.dim
        self.obs_type = self.obs_helper.type
        
        
class GymSpaceHelper():
    """ Helper functions to extract useful information from Gym spaces. """
    
    def __init__(self, space, distrib=None):

        self.space = space
        self.distrib = distrib
        
        if isinstance( space, gym.spaces.Discrete ):
            self.type = 'discrete'
            self.dim  = (space.n,)
            self.n_vars = space.n
            self.n_dims = None
        elif isinstance( space, gym.spaces.Box ):
            self.type = 'continuous'
            self.dim = space.shape
            
            self.n_dims = space.shape[0]
            if self.distrib is None:
                self.n_vars = None             
            elif self.distrib == 'normal':
                self.n_vars = self.n_dims * (self.n_dims + 3) // 2
            elif self.distrib in [ 'dirichlet', 'beta' ]:
                self.n_vars = self.n_dims + 1
            else:
                raise ValueError( 'Unknown distribution type: {}'.format(self.distrib) )
        else:
            raise ValueError( 'Unsupported gym space: {}'.format(space.__clas__) )
        
    def sample(self, params):
        distrib = self.get_distributions(params)        
        samples = distrib.sample()
        
        # Clip the samples to make sure they are contained within the appropriate range        
        if 'continuous' == self.type:
            if self.distrib == 'normal':
                samples = tf.clip_by_value( samples, self.space.low, self.space.high)
            elif self.distrib == 'beta':
                if self.n_dims == 1:
                    samples = self.space.low + (self.space.high - self.space.low) * samples[:,0]
                else:
                    raise ValueError( 'Beta distribution is only supported for dimensions == 1.' )
            
        # Ensure that the samples are 2-d tensors - otherwise the gym environments cannot handle them
        if self.dim[0] == 1:
            samples = tf.expand_dims(samples, 1)
        return samples.numpy()
    
    def get_distributions(self, distrib_params):
        if 'discrete' ==  self.type:
            return tfp.distributions.Categorical(logits=distrib_params)
        elif 'continuous' ==  self.type:            
            params = self.extract_params(distrib_params)            
            if 'normal' == self.distrib:
                mu = params['mu']
                if self.dim[0] == 1:                
                    sigma = params['sigma']
                    return tfp.distributions.Normal(mu, sigma)
                else:
                    cov = params['cov']                   
                    return tfp.distributions.MultivariateNormalFullCovariance(mu, cov)
            elif 'beta' == self.distrib:
                return tfp.distributions.Beta(*params['alpha'] )
            else:
                raise ValueError('Unsupported distribution: {}'.format(self.distrib))            
        else:
            raise ValueError( 'Unsupported space type: {}'.format(self.type) )                
            
    def extract_params(self, distrib_params):
        
        if 'continuous' !=  self.type:
            raise ValueError( 'Currently only supported for continuous spaces.' )
        
        assert distrib_params[0].shape == (self.n_vars,), \
                        "Dimensional mismatch between distrib_params and variable dimensions."
        if self.distrib == 'normal':
            if self.n_dims == 1:
                mu = distrib_params[:,0]                
                sigma = tf.nn.elu(distrib_params[:,1]) + 1e-6
                return dict(mu=mu, sigma=sigma )
            else:
                mu = distrib_params[:,:self.n_dims]
                LowTri = tfp.math.fill_triangular(distrib_params[:,self.n_dims:])
                cov = tf.matmul( tf.transpose(LowTri), LowTri )
                return dict(mu=mu, cov=cov )
        elif 'beta' == self.distrib:
            alpha = tf.nn.softplus(distrib_params)
            return dict(alpha=alpha)
        else:
            raise ValueError( \
                   'Unsupported space distribution: {}'.format(self.distrib))            
    
    
def make_env(environ_name, max_steps=None):
    """Function to create a new gym environment."""
    
    env = gym.make(environ_name)
    if max_steps is not None:
        env._max_episode_steps = max_steps
    return env    

def ewma(x, span=100): 
    return DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values

def get_mlp_layers( env, mlp_dims=(40,40), action_space_distr=None, activation='elu' ):
    """ Create a default MLP model for the actor-critic. 
        Input 'env' can be either the environment name (string), or an 
        instance of a Gym environment class. """    
    env_helper = EnvHelper( env, action_space_distr=action_space_distr)
    x = Input(env_helper.obs_dim)
    # Create the dense layers - # of layers and size of layers is based on the mlp_dims parameter
    dns = x
    for hidden_layer_size in mlp_dims:
        dns = Dense(hidden_layer_size, activation=activation )(dns)
    return x, dns

def get_mlp_actor( env, mlp_dims=(20,20), action_space_distr=None, activation='elu' ):
    env_helper = EnvHelper( env, action_space_distr=action_space_distr)    
    x, dns = get_mlp_layers( env, mlp_dims=mlp_dims, activation=activation, \
                            action_space_distr=action_space_distr )
    distrib_params = Dense( env_helper.n_action_vars, activation='linear', name='distrib_params')(dns)
    return Model( inputs=x, outputs=[ distrib_params ] )

def get_mlp_critic( env, mlp_dims=(40,40), activation='elu' ):
    x, dns = get_mlp_layers( env, mlp_dims=mlp_dims, activation=activation )
    V = Dense(1, activation='linear', name='V')(dns)
    return Model( inputs=x, outputs=[ V ] )

