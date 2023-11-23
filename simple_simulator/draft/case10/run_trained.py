import os, sys
expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'."+
             f" \nPlease activate the correct environment and try again.\nconda activate '{expected_env}'")

import tensorflow as tf
from environment import BouncingBallEnv
import numpy as np

# Function to calculate discounted rewards
def discount_rewards(rewards, gamma=0.99):
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

filepath="model_output"
loaded_model = tf.keras.models.load_model(filepath)
f = open(filepath+"/initial_angle","r")
init_angle=float(f.read())
f.close()
env = BouncingBallEnv()
state = env.reset(False, init_angle)
done = False
cumulative_reward = 0
episode_rewards = []

while not done:
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    
    # Predict action based on the current state using the loaded model
    action = loaded_model(state_tensor)
    
    # Step the environment
    state, reward, done, info = env.step(action.numpy())

    # Update the cumulative reward
    episode_rewards.append(reward)

    # Optionally render the environment
    env.render("video")

env.save_video("video.mp4")
episode_rewards = discount_rewards(episode_rewards)
print(f"Rewards sum: {np.sum(episode_rewards)} Init angle: {init_angle}")
