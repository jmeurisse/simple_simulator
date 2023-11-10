import os, sys
expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'."+
             f" \nPlease activate the correct environment and try again.\nconda activate '{expected_env}'")

import gym, imageio
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import mujoco

# Define the MuJoCo Gym environment
class BouncingBallEnv(gym.Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32) # same size than state

        # Load the model from an XML file
        self.model = mujoco.MjModel.from_xml_path("bouncing_ball.xml")
        self.data = mujoco.MjData(self.model)

    def step(self, action):
        # Apply action
        # print("action=",action)
        # print("data.ctrl=",self.data.ctrl)
        # print("action[0][0]=",action[0][0])
        # print("data.ctrl[0]=",self.data.ctrl[0])
        # print("train var=",model.trainable_variables) # 8 inputs and 64 nodes, 3 layers, weight = 8 x 64, bias = 64 
        self.data.ctrl[0] = action[0][0]
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Get the state
        xpos = self.data.qpos[:]
        xvel = self.data.qvel[:]
        # print("len(xpos)=",len(xpos)) # 1 value (slide position) + 4 values (bar 1 joint + ball 2 joints 1 hinge)
        # print("len(xvel)=",len(xvel)) # 1 value (slide position) + 4 values (bar 1 joint + ball 2 joints 1 hinge)
        state = np.concatenate([xpos, xvel])

        # Define reward and done condition
        reward = 1.0 if self._ball_is_on_bar(xpos) else 0
              
        # Define done condition
        ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')
        ball_z_position = xpos[2] + self.data.geom_xpos[ball_geom_id][2] # z ball position
        floor_z_position = 0.5  # between the height of the bar and the height of the floor
        done = ball_z_position <= floor_z_position

        return state, reward, done, {}

    def reset(self):
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)
        xpos = self.data.qpos[:]
        xvel = self.data.qvel[:]
        state = np.concatenate([xpos, xvel])
        # state = self.data.qpos.flatten().copy()
        state[3] = np.deg2rad(45)
        return state

    def _ball_is_on_bar(self, xpos):
        # Let's assume the bar's position and tolerance
        bar_z_position = 1.0  # The fixed z-position of the bar
        z_tolerance = 0.1     # How close the ball needs to be in the z-axis
        bar_x_min = -0.25      # The minimum x-position of the bar
        bar_x_max = 0.25       # The maximum x-position of the bar

        ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')

        # Check if the ball's x-position is within the range of the bar
        is_within_x_range = bar_x_min <= xpos[1] + self.data.geom_xpos[ball_geom_id][0] <= bar_x_max
        # Check if the ball's z-position is close to the bar's y-position
        is_close_to_z_position = abs(xpos[2] + self.data.geom_xpos[ball_geom_id][2] - bar_z_position) <= z_tolerance

        # The ball is considered to be 'on' the bar if both conditions are True
        # print("is_within_x_range=",is_within_x_range)
        # print("is_close_to_z_position=",is_close_to_z_position)
        return is_within_x_range and is_close_to_z_position
        
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

for episode in range(10):
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

filepath="model_output"
model.save(filepath)