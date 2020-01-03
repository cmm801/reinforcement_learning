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

from abc import ABC, abstractmethod
import distributions

DEFAULT_DISTRIBUTION = distributions.NormalDiag

class BaseModelAC(ABC):
    def __init__(self, model=None, feature_func=lambda x : x):
        self.model = model
        self.feature_func = feature_func
        
    def predict(self, state):
        features = self._extract_features(state)
        pre_processed = self._preprocess(features)
        model_outputs = self.model(pre_processed)
        post_processed = self._postprocess(model_outputs)
        return post_processed
        
    def _extract_features(self, state):
        return self.feature_func(np.atleast_2d(state))
        
    def _preprocess(self, state):
        return state

    def _postprocess(self, res):
        return res
    
    @abstractmethod
    def get_trainable_variables(self):
        pass
                    

class MLPModel(BaseModelAC):
    def __init__(self, model=None, feature_func=lambda x : x, env=None, scale_model_inputs=True):
        super().__init__(model=model, feature_func=feature_func)
        self.env = env
        self.scale_model_inputs = scale_model_inputs        
        self._init_standard_scaler()

    def _init_standard_scaler(self):
        if self.scale_model_inputs:
            sample_obs = np.array(
                [self.env.observation_space.sample() for x in range(10000)])
            sample_features = self._extract_features(sample_obs)
            self._standard_scaler = StandardScaler()
            self._standard_scaler.fit(sample_features)
        else:
            self._standard_scaler = None    
            
    def _preprocess(self, states):
        if self.scale_model_inputs:
            return self._standard_scaler.transform(states)
        else:
            return states
    
    def get_trainable_variables(self):
        return self.model.trainable_variables

    
class BaseAgent(ABC):
    def __init__(self, model=None, env=None, **kwargs):
        self.env = env                    
        if model is None:
            self.model = self._get_default_model(**kwargs)
        else:
            self.model = model            

    def get_trainable_variables(self):
        return self.model.get_trainable_variables()

    @abstractmethod
    def _get_default_model(self, **kwargs):
        pass


class BaseCritic(BaseAgent):
    def __init__(self, model=None, env=None, **kwargs):
        super().__init__(model=model, env=env)
        self._env_helper = EnvHelper(env)
        
    def get_state_values(self, states):
        state_value_raw = self.model.predict(states)
        return tf.reshape( state_value_raw, shape=(-1,) )
    
    
class BaseActor(BaseAgent):
    def __init__(self, model=None, env=None, action_space_distrib=DEFAULT_DISTRIBUTION, **kwargs):
        super().__init__(model=model, env=env)
        self._env_helper = EnvHelper(env, action_space_distrib=action_space_distrib)        
    
    def sample_actions(self, states ):
        """pick actions given numeric agent outputs (np arrays)"""
        raw_distrib_params = self.model.predict(states)
        return self._env_helper.sample_actions(raw_distrib_params)

    def get_action_distributions(self, states):
        raw_distrib_params = self.model.predict(states)
        return self._env_helper.get_action_distributions(raw_distrib_params)
    
    
class CriticMLP(BaseCritic):        
    def _get_default_model(self, scale_model_inputs=True, mlp_dims=(40,40), activation='elu' ):
        feature_dim = get_feature_dim(env=self.env, feature_func=feature_func)
        mlp = get_mlp_critic( obs_dim=feature_dim, mlp_dims=mlp_dims, activation=activation )
        return MLPModel( model=mlp, env=self.env, scale_model_inputs=scale_model_inputs)    

        
class ActorMLP(BaseActor):    
    def _get_default_model(self, scale_model_inputs=True, mlp_dims=(40,40), activation='elu' ):
        feature_dim = get_feature_dim(env=self.env, feature_func=feature_func)
        mlp = get_mlp_actor( obs_dim=feature_dim,
                            n_action_vars=self._env_helper.n_action_vars, 
                            mlp_dims=mlp_dims, activation=activation )
        return MLPModel( model=mlp, env=self.env, scale_model_inputs=scale_model_inputs)    
    
    
class CriticLinear(BaseCritic):        
    def _get_default_model(self, feature_func=None, scale_model_inputs=True, activation='elu' ):
        feature_dim = get_feature_dim(env=self.env, feature_func=feature_func)
        mlp = get_mlp_critic( obs_dim=feature_dim, mlp_dims=(), activation=activation )
        return MLPModel( model=mlp, feature_func=feature_func, 
                            env=self.env, scale_model_inputs=scale_model_inputs)    

        
class ActorLinear(BaseActor):    
    def _get_default_model(self, feature_func=None, scale_model_inputs=True, activation='elu' ):
        feature_dim = get_feature_dim(env=self.env, feature_func=feature_func)
        mlp = get_mlp_actor( obs_dim=feature_dim,
                                n_action_vars=self._env_helper.n_action_vars,
                                mlp_dims=(), activation=activation )
        return MLPModel( model=mlp, feature_func=feature_func,
                                env=self.env, scale_model_inputs=scale_model_inputs)
        
    
class ActorCritic():
    def __init__(self,
                 env,
                 entropy_coef=0.01,
                 reward_scale=1.,
                 actor=None, 
                 critic=None,
                 actor_learning_rate=0.001,
                 critic_learning_rate=0.001
                ):

        # Set parameters
        if isinstance( env, EnvBatch ):
            self.env = env
        else:
            self.env = EnvBatch(init_fun=env)
        
        # Set parameters
        self.entropy_coef = entropy_coef    # The coefficient of entropy used in the loss function
        self.reward_scale = reward_scale    # rescale rewards to keep gradients from exploding
        
        # Set agents
        self.actor = actor if actor is not None else ActorMLP(env=env)
        self.critic = critic if critic is not None else CriticMLP(env=env)        
        
        # Set optimizers
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=actor_learning_rate)        
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=critic_learning_rate)

    
    def train_step(self, states, next_states, actions, rewards, is_done, gamma=0.99):
        """Train the neural network from the states, rewards and transitions."""
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.actor.get_trainable_variables() )
            tape.watch(self.critic.get_trainable_variables() )
            
            # Get the probability distribution over possible actions, conditional on the state
            distrib = self.actor.get_action_distributions(states)

            # Get the estimated value of the current and next state
            state_values = self.critic.get_state_values(states)       
            next_state_values = self.critic.get_state_values(next_states)        
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
            
        grads_actor = tape.gradient( actor_loss, self.actor.get_trainable_variables() )
        grads_critic = tape.gradient( critic_loss, self.critic.get_trainable_variables() )
        del tape

        # Apply gradients      
        self.actor_optimizer.apply_gradients( zip( grads_actor, self.actor.get_trainable_variables() ) ) 
        self.critic_optimizer.apply_gradients( zip( grads_critic, self.critic.get_trainable_variables() ) )        
        
        return actor_loss, critic_loss, entropy
            
    def play_games(self, n_games=1):
        """Plays an a game from start till done, returns per-game rewards """

        # Create a temporary environment to play some games
        tmp_env = self.env.init_fun()
        
        game_rewards = []
        for _ in range(n_games):
            state = tmp_env.reset()

            total_reward = 0
            while True:
                action = self.actor.sample_actions(state[np.newaxis,:])
                state, reward, done, info = tmp_env.step(action[0])
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
            batch_actions = self.actor.sample_actions(batch_states)
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
    
    
class EnvBatch():
    """A class that allows us to play several games simultaneously, which improves the 
            convergence properties of A2C."""
       
    def __init__(self, init_fun, n_envs = 1):
        """ Create n_envs different environments for the possibility of asynchronous training. """
        self.init_fun = init_fun
        self.n_envs = n_envs
        self.envs = [ init_fun() for _ in range(n_envs)] 
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space        
        
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
    def __init__(self, env, action_space_distrib=distributions.NormalDiag ):
        if isinstance(env, str):
            env = make_env(env)
        elif isinstance(env, EnvBatch):
            env = env.envs[0]
            
        self._set_action_space(env.action_space, distrib=action_space_distrib)
        self._set_observation_space(env.observation_space)

    def sample_actions(self, raw_distrib_params):
        return self.action_helper.sample(raw_distrib_params)              
        
    def get_action_distributions(self, distrib_params):
        return self.action_helper.get_distributions(distrib_params)
                
    def _set_action_space(self, action_space, distrib):
        self.action_helper = ActionSpaceHelper(space=action_space, distrib=distrib)
        self.action_dim = self.action_helper.dim
        self.action_type = self.action_helper.type
        self.n_action_vars = self.action_helper.n_vars
             
    def _set_observation_space(self, obs_space):
        self.obs_helper = ObservationSpaceHelper(space=obs_space)
        self.obs_dim = self.obs_helper.dim
        self.obs_type = self.obs_helper.type


class GymSpaceHelper(ABC):
    """ Helper functions to extract useful information from Gym spaces. """    
    def __init__(self, space):
        self.space = space        
        if isinstance( space, gym.spaces.Discrete ):
            self.type = 'discrete'
            self.dim  = (space.n,)
            self.n_dims = space.n
        elif isinstance( space, gym.spaces.Box ):
            self.type = 'continuous'
            self.dim = space.shape            
            self.n_dims = space.shape[0]
        else:
            raise ValueError( 'Unsupported gym space: {}'.format(space.__clas__) )
            
            
class ObservationSpaceHelper(GymSpaceHelper):
    pass


class ActionSpaceHelper(GymSpaceHelper):
    def __init__(self, space, distrib):
        super().__init__(self, space=space)
        self._distrib = distrib(dim=self.n_dims)
        self.n_vars = self._distrib.get_number_of_params(self.n_dims)
        
    def get_distributions(self, distrib_params):
        self._distrib.set_distribution_from_array(raw_distrib_params)
        return self._distrib
        
    def sample(self, raw_distrib_params, n_samples=1):
        self._distrib.set_distribution_from_array(raw_distrib_params)
        samples = self._distrib.sample(n_samples=n_samples)
        
        # Clip the samples to make sure they are contained within the appropriate range
        if 'continuous' == self.type:
            if self.distrib == 'normal':
                samples = tf.clip_by_value( samples, self.space.low, self.space.high)
            elif self.distrib == 'beta':
                if self.n_dims == 1:
                    samples = self.space.low + (self.space.high - self.space.low) * samples[:,0]
                else:
                    raise ValueError( 'Beta distribution is only supported for dimensions == 1.' )
            elif self.distrib == 'multi-beta':
                 samples = self.space.low + (self.space.high - self.space.low) * samples
            else:
                raise ValueError( 'Unsupported distribution type: {}'.format(self.distrib) )
            
        # Ensure that the samples are 2-d tensors - otherwise the gym environments cannot handle them
        if self.dim[0] == 1:
            samples = tf.expand_dims(samples, 1)
        samples = samples.numpy()
        if self.type == 'discrete':
            return list(samples)
        else:
            return samples


def make_env(environ_name, max_steps=None):
    """Function to create a new gym environment."""
    
    env = gym.make(environ_name)
    if max_steps is not None:
        env._max_episode_steps = max_steps
    return env    

def ewma(x, span=100): 
    return DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values

def get_mlp_layers( obs_dim, mlp_dims=(40,40), activation='elu' ):
    """ Create a default MLP model for the actor-critic. """    
    x = Input(obs_dim)
    dns = x
    for hidden_layer_size in mlp_dims:
        dns = Dense(hidden_layer_size, activation=activation )(dns)
    return x, dns

def get_mlp_actor( obs_dim, n_action_vars, mlp_dims=(40,40), activation='elu' ):
    x, dns = get_mlp_layers( obs_dim, mlp_dims=mlp_dims, activation=activation )
    distrib_params = Dense( n_action_vars, activation='linear', name='distrib_params')(dns)
    return Model( inputs=x, outputs=[ distrib_params ] )

def get_mlp_critic( obs_dim, mlp_dims=(40,40), activation='elu' ):
    x, dns = get_mlp_layers( obs_dim, mlp_dims=mlp_dims, activation=activation )
    V = Dense(1, activation='linear', name='V')(dns)
    return Model( inputs=x, outputs=[ V ] )

def get_feature_dim(env, feature_func):
    sample_obs = np.atleast_2d( env.observation_space.sample() )
    return feature_func(sample_obs).shape 
