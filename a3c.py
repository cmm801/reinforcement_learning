import numpy as np
import tensorflow as tf
import gym

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

# Use float32 by default
tf.keras.backend.set_floatx('float32')

class Agent:
    def __init__(self, env, reward_scale=1., model=None):
        """A simple actor-critic agent"""

        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape

        # Optionally rescale the rewards when it can help keep gradients from exploding
        self.reward_scale = reward_scale

        # Initialize the optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001, epsilon=1e-8)

        if model is None:
            model = self._get_default_model()

        self.set_model(model)

    def _get_default_model(self):
        """Create a default model for the actor critic.
        """
        n_hidden_0 = n_hidden_1 = 20            
        x = Input(self.state_dim)
        dns_1 = Dense(n_hidden_0, activation='relu')(x)
        dns_2 = Dense(n_hidden_1, activation='relu')(dns_1)
        V = Dense(1, activation='linear', name='V')(dns_2)
        logits = Dense( self.n_actions, activation='linear', name='logits')(dns_2)
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
        logits, state_values = agent_outputs
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in policy])

    def train_step(self, states, next_states, actions, rewards, is_done, gamma=0.99):
        """Train the neural network from the states, rewards and transitions."""
        
        with tf.GradientTape() as tape:
            tape.watch(self.model.weights)
            loss, ent_t = self.calc_loss( states, next_states, actions, rewards, is_done, gamma=gamma )

        # Calculate the gradient of the objective
        grads = tape.gradient( loss, self.model.weights )
        self.optimizer.apply_gradients( zip( grads, self.model.weights ) )
        return loss, ent_t

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
        actor_loss = -tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - 0.01 * tf.reduce_mean(entropy)

        # Critic loss calculation
        target_state_values = rewards + gamma * next_state_values
        critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2 )
        return actor_loss + critic_loss, entropy        
        
    def play_games(self, env, n_games=1):
        """Plays an a game from start till done, returns per-game rewards """

        game_rewards = []
        for _ in range(n_games):
            state = env.reset()

            total_reward = 0
            while True:
                state = state[:,np.newaxis].T
                action = self.sample_actions(self.step([state]))[0]
                state, reward, done, info = env.step(action)
                total_reward += reward
                if done: break

            # We rescale the reward back to ensure compatibility
            # with other evaluations.
            game_rewards.append(total_reward / self.reward_scale)
        return game_rewards        
        
class EnvBatch():
    """A class that allows us to play several games simultaneously, which improves the convergence properties of A2C."""
    
    def __init__(self, environ_name, n_envs = 10, max_steps=None):
        """ Create n_envs environments """
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