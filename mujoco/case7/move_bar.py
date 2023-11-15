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

# Create the model
model = PolicyModel()

# Function to calculate discounted rewards
def discount_rewards(rewards, gamma=0.99):
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

max_rewards = -1
n_episodes = 10

init_angles=[45,-45]
init_actions=[50,-50]
frames=[]
for i in range(0,len(init_angles)):
    with tf.GradientTape() as tape:
        env = BouncingBallEnv(init_angles[i],init_actions[i],frames)
        state = env.reset()
        done = False

        while not done:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action = model(state_tensor)
            state, reward, done, _ = env.step(action.numpy())
            env.render("video")
    
        # for frame in env.frames:
        #     frames.append(frame)
        if i == len(init_angles)-1:
            print("Create move_bar.mp4")
            env.save_video("move_bar.mp4") 