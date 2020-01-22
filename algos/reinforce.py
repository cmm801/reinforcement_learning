import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Reinforce():

    def __init__(self, env, model=None, optimizer=None):
        
        self.env = env
        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape
        
        if model is not None:
            self.model = model
        else:
            n_hidden_0 = n_hidden_1 = 20            
            self.model = Sequential( [
                        Dense( n_hidden_0, activation='relu', input_shape=self.state_dim ),
                        Dense( n_hidden_1, activation='relu' ),    
                        Dense( self.n_actions,  activation='linear' ),
            ])            
            
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.optimizers.Adam(learning_rate=0.001, epsilon=1e-8)

    def calc_cum_rewards(self, rewards, gamma ):    
        #cum_rewards = get_cumulative_rewards(_rewards=rewards, _gamma=gamma )    
        N = rewards.shape[0]

        cum_rewards_list = [np.nan] * N
        cum_rewards_list[-1] = rewards[-1]
        for t in range(1, rewards.shape[0] ):
            cum_rewards_list[-(t+1)] = rewards[-(t+1)] + gamma * cum_rewards_list[-t] 

        cum_rewards = tf.reshape( tf.convert_to_tensor( cum_rewards_list, dtype=tf.float32 ), (N,) )
        return cum_rewards

    def train_step(self, states, actions, cum_rewards, gamma, entropy_weight ):
        """given full session, trains agent with policy gradient"""

        with tf.GradientTape() as tape:
            tape.watch(self.model.weights)

            logits = self.model(states)
            policy = tf.nn.softmax(logits)
            log_policy = tf.nn.log_softmax(logits)
            entropy_vec = tf.reduce_sum( -policy * tf.math.log(policy), axis=1 )
            entropy = tf.reduce_mean( entropy_vec )

            N = policy.shape[0]
            indices = tf.stack( [ tf.range(N), tf.reshape( actions, (N,) ) ], axis=-1)
            log_probs_for_actions = tf.gather_nd( log_policy, indices)

            J = tf.reduce_mean( log_probs_for_actions * cum_rewards )           

            current_loss = -J - entropy_weight * entropy

        # Calculate the gradient of the objective
        grads = tape.gradient( current_loss, self.model.weights )
        self.optimizer.apply_gradients( zip( grads, self.model.weights ) )

        return current_loss

    def generate_session(self, t_max, gamma, entropy_weight ):
        """play env with REINFORCE agent and train at the session end"""

        # arrays to record session
        S, A, R = [], [], []

        obs = self.env.reset()
        obs = obs[:,np.newaxis].T

        for t in range(t_max):

            # action probabilities array aka \pi(a|s)     
            logits = self.model(obs)
            action = tf.random.categorical( logits, num_samples=1 )

            # Move to next state
            new_obs, r, done, info = self.env.step( action.numpy()[0,0] )
            new_obs = new_obs[:,np.newaxis].T

            # record session history to train later
            S.append(obs)
            A.append(action)
            R.append(r)

            obs = new_obs
            if done:
                break

        # Form Tensors out of the collected outputs
        state_mtx = tf.stack( np.vstack( S ) )
        action_mtx = tf.dtypes.cast( tf.stack( np.vstack( A ) ), tf.int32 )
        reward_mtx = tf.stack( np.vstack( R ) )

        cum_reward_mtx = self.calc_cum_rewards( reward_mtx, gamma )
        curr_loss = self.train_step( states=state_mtx, actions=action_mtx,
                       cum_rewards=cum_reward_mtx, gamma=gamma, entropy_weight=entropy_weight )

        # return session rewards to print them later
        return sum(R)    
    
    def train(self, n_steps=20, t_max = 1000, gamma = 0.99, entropy_weight = 0.10 ):
        for i in range(n_steps):
            rewards = []
            for _ in range(100):
                new_rwd = self.generate_session( t_max=t_max, gamma=gamma, entropy_weight=entropy_weight)
                rewards.append(new_rwd)

            print("mean reward:%.3f" % (np.mean(rewards)))

