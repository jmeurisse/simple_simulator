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
#model = PolicyModel()
# load model
filepath="model_output"
model = tf.keras.models.load_model(filepath)

# Training loop
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# Function to calculate discounted rewards
def discount_rewards(rewards, gamma=0.99):
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

filepath="model_output"
file_max=filepath+"/max_rewards"
if os.path.exists(file_max):
    with open(file_max, 'r') as file:
        number_str = file.read().strip()  # Read and strip whitespace
        number = float(number_str)  # Convert to float
else:
    number=0
max_rewards = number
print("max_rewards = ",max_rewards)
n_episodes = int(20)
max_frames=[]
init_angle = 45

for episode in range(n_episodes):
    with tf.GradientTape() as tape:
        state = env.reset(False, init_angle)
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

        # likelihood that an action will lead to a higher reward given a given output of the model

        # Policy gradient loss
        # The loss is the negative of the mean of these weighted log probabilities
        # Negative sign is because we want to maximize rewards, but TensorFlow performs minimization
        loss = -tf.reduce_mean(weighted_log_probs)

        # Compute the gradients
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        rewards_sum = np.sum(episode_rewards)
        print(f"Episode: {episode} Reward: {rewards_sum} Loss: {loss} Initial angle: {init_angle}")

        if rewards_sum > max_rewards:
            max_rewards=rewards_sum
            max_model=model
            max_frames=env.frames
            max_env = env
            max_episode = episode
            print("max is episode",episode)

        # env.frames=[]

import matplotlib.pyplot as plt
env=max_env
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(env.rewards_direction, label='rewards_direction', ls="dashed", color='tab:blue')
ax1.plot(env.rewards_wall, label='rewards_wall', ls="dashed", color='tab:cyan')
ax1.plot(env.rewards_range_pos, label='rewards_range_pos', ls="dashed", color='tab:green')
ax1.plot(env.rewards_range_vel, label='rewards_range_vel', ls="dashed", color='black')
ax1.set_ylim(-0.5,1)
ax1.set_xlabel('episode')
ax1.set_ylabel('rewards')
ax1.tick_params(axis='y')

# Create a second y-axis (for losses)
ax2 = ax1.twinx()
ax2.plot(env.ball_center_x_pos_table, label='ball x', ls="solid", color='tab:orange')
ax2.plot(env.ball_center_z_pos_table, label='ball z', ls="solid", color='tab:brown')
ax2.plot(env.bar_center_x_pos_table, label='bar x', ls="solid", color='tab:red')
ax2.set_ylabel('position', color='tab:red')
ax2.tick_params(axis='y')
ax2.set_ylim(-5,2)

# Add a legend for all y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc=1)

# Set the title and display the plot
plt.title('Episode ' + str(max_episode) +' rewards')
plt.grid(True)
plt.savefig('max_rewards_'+ str(max_episode)+'.png')




#os.system("rm -rf episodes.mp4")
#print("Create episode.mp4")
#env.save_video("episodes.mp4")
max_model.save(filepath)
f=open(filepath+"/max_rewards","w")
f.write(str(max_rewards))
f.close()
