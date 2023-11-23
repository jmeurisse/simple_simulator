import os, sys
expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'."+
             f" \nPlease activate the correct environment and try again.\nconda activate '{expected_env}'")

from gym_envir import BouncingBallEnv
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# Define the policy model using TensorFlow
class PolicyModel(tf.keras.Model):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu', trainable=True)
        self.dense2 = layers.Dense(64, activation='relu', trainable=True)
        self.dense3 = layers.Dense(1, activation='tanh', trainable=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# Create the environment and the model
env = BouncingBallEnv()
model = PolicyModel()

# Training loop
optimizer = tf.optimizers.Adam(learning_rate=0.01)

# Function to calculate discounted rewards
def discount_rewards(rewards, gamma=0.99):
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

min_loss = sys.float_info.max

for episode in range(100):
    with tf.GradientTape() as tape:
        state = env.reset()
        episode_states, episode_actions, episode_rewards = [], [], []
        done = False

        while not done:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action = model(state_tensor)
            
            state, reward, done, _ = env.step(action.numpy())
            episode_rewards.append(reward)
            episode_states.append(state_tensor)
            episode_actions.append(action)

        # Prepare the data for loss calculation
        episode_states = tf.concat(episode_states, axis=0)
        episode_actions = tf.concat(episode_actions, axis=0)
        episode_rewards = discount_rewards(episode_rewards)
        
        # Calculate loss
        logits = model(episode_states)
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=episode_actions)
        loss = tf.reduce_mean(neg_log_prob * episode_rewards)

        # Compute the gradients
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Apply the gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Episode: {episode} Reward: {np.sum(episode_rewards)}")

        if loss < min_loss:
            min_loss=loss
            min_model=model

filepath="model_output"
min_model.save(filepath)