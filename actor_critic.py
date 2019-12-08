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
                 lr_actor=0.001,
                 lr_critic=0.001,
                 model=None, 
                 action_space_distr='normal', 
                 scale_model_inputs=True
                ):

        # Set parameters
        if isinstance( env, EnvBatch ):
            self.env = env
        else:
            env_name = env.unwrapped.spec.id
            self.env = EnvBatch(env_name)

        # Initialize the agent
        self.agent = Agent(
                            env=self.env.envs[0],
                            entropy_coef=entropy_coef,
                            reward_scale=reward_scale,
                            lr_actor=lr_actor,
                            lr_critic=lr_critic,
                            model=model,
                            action_space_distr=action_space_distr,
                            scale_model_inputs=scale_model_inputs
                           )
        
    def play_games(self, n_games=1):
        """Plays an a game from start till done, returns per-game rewards """

        # Create a temporary environment to play some games
        tmp_env = make_env( self.env.environ_name, max_steps=self.env.max_steps)
        
        game_rewards = []
        for _ in range(n_games):
            state = tmp_env.reset()

            total_reward = 0
            while True:
                logits, _ = self.agent.step(state[np.newaxis,:])
                action = self.agent.sample_actions(logits)[0]
                state, reward, done, info = tmp_env.step(action)
                total_reward += reward
                if done: break

            # We rescale the reward back to ensure compatibility
            # with other evaluations.
            game_rewards.append(total_reward / self.agent.reward_scale)
        return game_rewards       
    
    def run(self, n_iters=2000, gamma=0.99, n_games_per_plot_update=3, n_iters_per_plot_update=100 ):
        """Run the training for a number of iterations, and regularly plot the progress."""

        # Get the initial states
        batch_states = self.env.reset()
        rewards_mean, rewards_max, rewards_min = [], [], []
        entropy_history = []        

        for i in trange(n_iters):
            logits, _ = self.agent.step(batch_states)
            batch_actions = self.agent.sample_actions(logits)
            batch_next_states, batch_rewards, batch_done, _ = self.env.step(batch_actions)

            # Train the neural network from the states, rewards and transitions
            actor_loss, critic_loss, ent_t = self.agent.train_step( batch_states, batch_next_states, \
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
           
    
class Agent():
    def __init__(self, 
                 env,
                 entropy_coef=0.01,
                 reward_scale=1.,
                 lr_actor = 0.001,
                 lr_critic = 0.001,
                 model=None,
                 action_space_distr='normal', 
                 scale_model_inputs=True
                ):
        """A simple actor-critic agent"""
        
        # Set parameters
        self.entropy_coef = entropy_coef    # The coefficient of entropy used in the loss function
        self.reward_scale = reward_scale    # rescale rewards to keep gradients from exploding
        
        # Set properties needed for normalizing the inputs before appling the NN model
        self.scale_model_inputs = scale_model_inputs
        self._initialize_scaling(env)
        
        # Extract action space / observation space information from the environment
        self._env_helper = EnvHelper(env, action_space_distr=action_space_distr)
        self.obs_dim = self._env_helper.obs_dim
        self.n_action_vars = self._env_helper.n_action_vars

        # Initialize the optimizer
        self.optimizer_actor = tf.optimizers.Adam(learning_rate=lr_actor)
        self.optimizer_critic = tf.optimizers.Adam(learning_rate=lr_critic)

        # Set the model
        if model is None:
            model = get_mlp( env, action_space_distr=action_space_distr)
        self.set_model(model)        

    def set_model(self, model):
        """ Use the provided model, if it has the correct input/output dimensions. """
        assert model.input_shape == (None,) + self.obs_dim, \
                            'Input shape is incompatible with the chosen environment.'
        assert model.output_shape == [ (None,self.n_action_vars), (None,1) ], \
                            'Output shape is incompatible with the chosen environment.'
        self.model = model        
        
    def step(self, state_t):
        """Takes agent's previous step and observation, returns next state and 
                whatever it needs to learn (tf tensors)"""        
        scaled_state = self._apply_scaling(state_t)
        logits, state_value_raw = self.model(state_t)        
        state_value = tf.reshape( state_value_raw, shape=(-1,) )        
        return logits, state_value
    
    def sample_actions(self, logits):
        """pick actions given numeric agent outputs (np arrays)"""
        return self._env_helper.sample_actions(logits)

    def get_action_distributions(self, logits):
        return self._env_helper.get_action_distributions(logits)
    
    def train_step(self, states, next_states, actions, rewards, is_done, gamma=0.99):
        """Train the neural network from the states, rewards and transitions."""
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            actor_loss, critic_loss, entropy = self.calc_loss( states, next_states, \
                                                         actions, rewards, is_done, gamma=gamma )
        grads_actor = tape.gradient( actor_loss, self.model.trainable_variables )
        grads_critic = tape.gradient( critic_loss, self.model.trainable_variables )
        del tape

        # Apply gradients
        self.optimizer_actor.apply_gradients( zip( grads_actor, self.model.trainable_variables ) ) 
        self.optimizer_critic.apply_gradients( zip( grads_critic, self.model.trainable_variables ) )        
        
        return actor_loss, critic_loss, entropy

    def calc_loss( self, states, next_states, actions, rewards, is_done, gamma=0.99 ):
        """Calculate the actor plus critic loss."""
        
        # logits[n_envs, n_actions] and state_values[n_envs, n_actions]
        logits, state_values = self.step(states)
        distrib = self.get_action_distributions(logits)
        next_logits, next_state_values = self.step(next_states)
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
        critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2 )
        return actor_loss, critic_loss, entropy 

    def _initialize_scaling(self, env):
        if self.scale_model_inputs:
            samples = np.array(
                [env.observation_space.sample() for x in range(10000)])
            self._standard_scaler = StandardScaler()
            self._standard_scaler.fit(samples)
        else:
            self._standard_scaler = None    
            
    def _apply_scaling(self, state):
        if self.scale_model_inputs:
            return self._standard_scaler.transform(state)
        else:
            return state

    
class EnvBatch():
    """A class that allows us to play several games simultaneously, which improves the 
            convergence properties of A2C."""
    
    def __init__(self, environ_name, n_envs = 1, max_steps=None):
        """ Create n_envs different environments for the possibility of asynchronous training. """
        self.environ_name = environ_name
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.envs = [make_env(environ_name, max_steps=max_steps) for _ in range(n_envs)]
        
    def reset(self):
        """ Reset all games and return [n_envs, *obs_shape] observations """
        return np.array([env.reset() for env in self.envs])
    
    def step(self, actions):
        """
        Send a vector[batch_size] of actions into respective environments
        :returns: observations[n_envs, *obs_shape], rewards[n_envs], done[n_envs,], info[n_envs]
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, done, info = map(np.array, zip(*results))
        
        # reset environments automatically
        for i in range(len(self.envs)):
            if done[i]:
                new_obs[i] = self.envs[i].reset()
        
        return new_obs, rewards, done, info   
    

class EnvHelper():
    
    def __init__(self, env, action_space_distr='normal'):                        
        self.action_space_distr = action_space_distr
        self._set_action_space(env.action_space, distrib=self.action_space_distr)
        self._set_observation_space(env.observation_space)

    def sample_actions(self, logits):
        return self.action_helper.sample(logits)
              
    def get_action_distributions(self, logits):
        return self.action_helper.get_distributions(logits)
                
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
        elif isinstance( space, gym.spaces.Box ):
            self.type = 'continuous'
            self.dim = space.shape
            
            N = space.shape[0]
            if self.distrib is None:
                self.n_vars = None             
            elif self.distrib == 'normal':
                self.n_vars = N * (N + 3) // 2
            else:
                raise ValueError( 'Unknown distribution type: {}'.format(self.distrib) )
        else:
            raise ValueError( 'Unsupported gym space: {}'.format(space.__clas__) )
        
    def sample(self, logits ):
        distrib = self.get_distributions(logits)
        samples = distrib.sample()
        
        # Clip the samples to make sure they are contained within the appropriate range        
        if 'continuous' == self.type:
            samples = tf.clip_by_value( samples, self.space.low, self.space.high)
            
        # Ensure that the samples are 2-d tensors - otherwise the gym environments cannot handle them
        if self.dim[0] == 1:
            samples = tf.expand_dims(samples, 1)
        return samples.numpy()
    
    def get_distributions(self, logits):
        if 'discrete' ==  self.type:
            return tfp.distributions.Categorical(logits=logits)
        elif 'continuous' ==  self.type:            
            params = self.extract_params(logits)            
            if 'normal' == self.distrib:
                mu = params['mu']
                cov = params['cov']
                if self.dim[0] == 1:                
                    return tfp.distributions.Normal(mu, cov)
                else:
                    return tfp.distributions.MultivariateNormalFullCovariance(mu, cov)
            else:
                raise ValueError('Unsupported distribution: {}'.format(self.distrib))            
        else:
            raise ValueError( 'Unsupported space type: {}'.format(self.type) )                
            
    def extract_params(self, logits):
        
        if 'continuous' !=  self.type:
            raise ValueError( 'Currently only supported for continuous spaces.' )
        
        N = self.dim[0]
        assert logits[0].shape == (self.n_vars,), \
                        "Dimensional mismatch between logits and variable dimensions."
        if self.distrib == 'normal':
            if N == 1:
                mu = logits[:,0]                
                cov = tf.nn.softplus(logits[:,1]) + 1e-6
            else:
                mu = logits[:,:N]                
                LowTri = tfp.math.fill_triangular(logits[:,N:])
                cov = tf.matmul( tf.transpose(LowTri), LowTri )
            return dict(mu=mu, cov=cov )
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


def get_mlp( env, mlp_dims=(20,20), action_space_distr=None ):
    """ Create a default MLP model for the actor-critic. 
        Input 'env' can be either the environment name (string), or an instance of a Gym environment class.
    """
    
    if isinstance(env, str):
        env_helper = EnvHelper( make_env(env), action_space_distr=action_space_distr)
    else:
        env_helper = EnvHelper( env, action_space_distr=action_space_distr)        
    
    x = Input(env_helper.obs_dim)

    # Create the dense layers - # of layers and size of layers is based on the mlp_dims parameter
    dns = x
    for hidden_layer_size in mlp_dims:
        dns = Dense(hidden_layer_size, activation='relu')(dns)

    # Split the final output into two components - the actor part (logits) and critic part (V)
    V = Dense(1, activation='linear', name='V')(dns)
    logits = Dense( env_helper.n_action_vars, activation='linear', name='logits')(dns)
    return Model( inputs=x, outputs=[ logits, V ] )