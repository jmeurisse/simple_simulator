import os, sys
expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'."+
             f" \nPlease activate the correct environment and try again.\nconda activate '{expected_env}'")

from environment import BouncingBallEnv
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow_probability as tfp


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
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

# Function to calculate discounted rewards
def discount_rewards(rewards, gamma=0.99):
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

max_rewards = -1
n_episodes = 1e3
max_frames=[]

for episode in range(n_episodes):
    with tf.GradientTape() as tape:
        state = env.reset()
        episode_states, episode_actions, episode_rewards = [], [], []
        done = False
        i = 0

        while not done:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            logits = model(state_tensor)
            state, reward, done, _ = env.step(logits.numpy())
            action = tf.constant([[env.data.ctrl[0]]], dtype=tf.float32) 
            episode_rewards.append(reward)
            episode_states.append(state_tensor)
            episode_actions.append(action)
            env.render("video")
            i+=1
    
        # Prepare the data for loss calculation
        episode_states = tf.concat(episode_states, axis=0)
        episode_actions = tf.concat(episode_actions, axis=0)
        episode_rewards = discount_rewards(episode_rewards)
        
        # Calculate loss
        logits = model(episode_states)
        # mse_loss_fun = tf.keras.losses.MeanSquaredError()
        # mse_loss = mse_loss_fun(episode_actions, logits) # Mean Square Error between the model's predicted actions and the actions taken. 
        # loss = tf.reduce_mean(mse_loss * episode_rewards)

        # Assuming your logits are the means of a Gaussian distribution for the actions
        # You may also learn the standard deviation, but for simplicity, let's use a fixed value here
        std_dev = 0.5  # Standard deviation for the Gaussian policy
        logits = model(episode_states)

        # Calculate the Gaussian log probabilities
        # Here, we compute the log probability of the episode_actions under the Gaussian distribution defined by logits (mean) and std_dev (standard deviation)
        action_prob_distribution = tfp.distributions.Normal(loc=logits, scale=std_dev)
        log_prob_actions = action_prob_distribution.log_prob(episode_actions)

        # Weight log probabilities by rewards
        # You can modify this step to include a baseline or use advantage instead of raw rewards for more sophisticated algorithms
        weighted_log_probs = log_prob_actions * episode_rewards

        # Policy gradient loss
        # The loss is the negative of the mean of these weighted log probabilities
        # Negative sign is because we want to maximize rewards, but TensorFlow performs minimization
        loss = -tf.reduce_mean(weighted_log_probs)


        # print("i=",i)
        # print("episode_states=",episode_states.shape)
        # print("episode_actions=",episode_actions)
        # print("logits=",logits)
        # print("action_prob_distribution=",action_prob_distribution)
        # print("log_prob_actions",log_prob_actions)
        # print("loss", loss)

        # Compute the gradients
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Apply the gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        rewards_sum = np.sum(episode_rewards)
        print(f"Episode: {episode} Reward: {rewards_sum} Loss: {loss} Initial angle: {env.random_integer}")

        if rewards_sum > max_rewards:
            max_rewards=rewards_sum
            max_model=model
            max_frames=env.frames
            max_env = env
            print("max is episode",episode)

        env.frames=[]

env.frames=max_frames
print("Create max_episode.mp4")
env.save_video("max_episodes.mp4") 
filepath="model_output"
max_model.save(filepath)
f = open(filepath+"/initial_angle","w")
f.write("{:.4e}".format(max_env.random_integer))
f.close()