import numpy as np
import tensorflow as tf
import gym
from pandas import DataFrame
from IPython.display import clear_output
from tqdm import trange
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Use float32 by default
tf.keras.backend.set_floatx('float32')

class ActorCritic():
    def __init__(self,
                 environ_name,
                 n_envs=10,                 
                 max_episode_steps=None,                 
                 mlp_dims=(20,20),
                 entropy_coef=0.01,
                 reward_scale=1.,
                 lr_actor=0.001,
                 lr_critic=0.001,
                 model=None                 
                ):

        # Set parameters
        self.env = EnvBatch(environ_name, n_envs=n_envs, max_steps=max_episode_steps )
        
        # Extract variables describing the environment        
        n_actions = self.env.envs[0].action_space.n
        state_dim = self.env.envs[0].observation_space.shape

        # Initialize the agent
        self.agent = Agent(
                            n_actions,
                            state_dim,
                            mlp_dims=(20,20),
                            entropy_coef=0.01,
                            reward_scale=1.,
                            lr_actor=0.001,
                            lr_critic=0.001,
                            model=None                 
                           )
        
    def play_games(self, n_games=1):
        """Plays an a game from start till done, returns per-game rewards """

        # Create a temporary environment to play some games
        tmp_env = make_env(self.env.environ_name, max_steps=self.env.max_steps)
        
        game_rewards = []
        for _ in range(n_games):
            state = tmp_env.reset()

            total_reward = 0
            while True:
                state = state[:,np.newaxis].T
                action = self.agent.sample_actions(self.agent.step([state]))[0]
                state, reward, done, info = tmp_env.step(action)
                total_reward += reward
                if done: break

            # We rescale the reward back to ensure compatibility
            # with other evaluations.
            game_rewards.append(total_reward / self.agent.reward_scale)
        return game_rewards       
    
    def run(self, n_iters=2000, gamma=0.99, n_games_per_plot_update=3):
        """Run the training for a number of iterations, and regularly plot the progress."""

        # Get the initial states
        batch_states = self.env.reset()
        rewards_history = []
        entropy_history = []        
        
        for i in trange(n_iters):
            batch_actions = self.agent.sample_actions(self.agent.step(batch_states))
            batch_next_states, batch_rewards, batch_done, _ = self.env.step(batch_actions)

            # Train the neural network from the states, rewards and transitions
            actor_loss, critic_loss, ent_t = self.agent.train_step( batch_states, batch_next_states, \
                                          batch_actions, batch_rewards, batch_done, gamma=gamma )
            batch_states = batch_next_states
            entropy_history.append(np.mean(ent_t))

            if i % 500 == 0:
                if i % 500 == 0:
                    avg_score = np.mean( self.play_games(n_games=n_games_per_plot_update) )
                    rewards_history.append(avg_score)

                clear_output(True)
                plt.figure(figsize=[8, 4])
                plt.subplot(1, 2, 1)
                plt.plot(rewards_history, label='rewards')
                plt.plot(ewma(np.array(rewards_history), span=10), marker='.', label='rewards ewma@10')
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
        return rewards_history, entropy_history
           
    
class Agent():
    def __init__(self, 
                 n_actions,
                 state_dim,
                 mlp_dims=(20,20),
                 entropy_coef=0.01,
                 reward_scale=1.,
                 lr_actor = 0.001,
                 lr_critic = 0.001,
                 model=None                 
                ):
        """A simple actor-critic agent"""
        
        # Set parameters
        self.mlp_dims = mlp_dims             # The dimensions used for the MLP if no model is provided        
        self.entropy_coef = entropy_coef    # The coefficient of entropy used in the loss function
        self.reward_scale = reward_scale    # rescale the rewards when it can help keep gradients from exploding        

        # Set problem size variables from the environment
        self.n_actions = n_actions
        self.state_dim = state_dim

        # Initialize the optimizer
        self.optimizer_actor = tf.optimizers.Adam(learning_rate=lr_actor)
        self.optimizer_critic = tf.optimizers.Adam(learning_rate=lr_critic)

        # Set the model
        if model is None:
            model = self._get_default_model()
        self.set_model(model)        
        
    def _get_default_model(self):
        """Create a default model for the actor critic.
        """
        x = Input(self.state_dim)
        
        # Create the dense layers - # of layers and size of layers is based on the mlp_dims parameter
        dns = x
        for hidden_layer_size in self.mlp_dims:
            dns = Dense(hidden_layer_size, activation='relu')(dns)

        # Split the final output into two components - the actor part (logits) and critic part (V)
        V = Dense(1, activation='linear', name='V')(dns)
        logits = Dense( self.n_actions, activation='linear', name='logits')(dns)
        return Model( inputs=x, outputs=[ logits, V ] )

    def set_model(self, model):
        """Check the model has the correct dimensions, and set it.
        """
        assert model.input_shape == (None,) + self.state_dim, \
                            'Input shape is incompatible with the chosen environment.'
        assert model.output_shape == [ (None,self.n_actions), (None,1) ], \
                            'Output shape is incompatible with the chosen environment.'
        self.model = model        
       
    def step(self, state_t):
        """Takes agent's previous step and observation, returns next state and whatever it needs to learn (tf tensors)"""        
        # Apply neural network
        logits, state_value_raw = self.model(state_t)        
        state_value = tf.reshape( state_value_raw, shape=(-1,) )        
        return (logits, state_value)
    
    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, _ = agent_outputs
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in policy])

    def train_step(self, states, next_states, actions, rewards, is_done, gamma=0.99):
        """Train the neural network from the states, rewards and transitions."""
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            actor_loss, critic_loss, entropy = self.calc_loss( states, next_states, actions, rewards, is_done, gamma=gamma )
            tot_loss = actor_loss + critic_loss
  
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
        next_logits, next_state_values = self.step(next_states)
        next_state_values = next_state_values * (1 - is_done)

        # probabilities and log-probabilities for all actions
        probs = tf.nn.softmax(logits)            # [n_envs, n_actions]
        logprobs = tf.nn.log_softmax(logits)     # [n_envs, n_actions]

        # log-probabilities only for agent's chosen actions
        logp_actions = tf.reduce_sum(logprobs * tf.one_hot(actions, self.n_actions), axis=-1) # [n_envs,]

        # Actor loss calculation (policy gradient approach)
        advantage = rewards + gamma * next_state_values - state_values        
        entropy = -tf.reduce_sum( tf.multiply( probs, tf.math.log(probs) ), axis=1 )
        actor_loss = -tf.reduce_mean( logp_actions * tf.stop_gradient(advantage) ) \
                     -self.entropy_coef * tf.reduce_mean(entropy)

        # Critic loss calculation
        target_state_values = rewards + gamma * next_state_values
        critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2 )
        return actor_loss, critic_loss, entropy 
        
class EnvBatch():
    """A class that allows us to play several games simultaneously, which improves the convergence properties of A2C."""
    
    def __init__(self, environ_name, n_envs = 10, max_steps=None):
        """ Create n_envs environments """
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
    
    
def make_env(environ_name, max_steps=None):
    """Function to create a new gym environment."""
    
    env = gym.make(environ_name)
    if max_steps is not None:
        env._max_episode_steps = max_steps
    return env    


def ewma(x, span=100): 
    return DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values
