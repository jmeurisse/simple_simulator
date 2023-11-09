import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import mujoco

# Define the MuJoCo Gym environment
class BouncingBallEnv(gym.Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32) # same size than state

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
        # print("len(xpos)=",len(xpos)) # 1 value (slide position) + 7 values (3D position + 4D quaternion)
        # print("len(xvel)=",len(xvel)) # 1 value (slide position) + 6 values (3D linear velocity + 3D angular velocity)
        state = np.concatenate([xpos, xvel])

        # Define reward and done condition
        reward = 1.0 if self._ball_is_on_bar(xpos) else -1.0
        done = False if reward > 0 else True

        return state, reward, done, {}

    def reset(self):
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)
        state = self.data.qpos.flatten().copy()
        return state

    def _ball_is_on_bar(self, xpos):
        # Let's assume the bar's position and tolerance
        bar_y_position = 1.0  # The fixed y-position of the bar
        y_tolerance = 0.1     # How close the ball needs to be in the y-axis
        bar_x_min = -1.0      # The minimum x-position of the bar
        bar_x_max = 1.0       # The maximum x-position of the bar

        # Check if the ball's x-position is within the range of the bar
        is_within_x_range = bar_x_min <= xpos[0] <= bar_x_max
        # Check if the ball's y-position is close to the bar's y-position
        is_close_to_y_position = abs(xpos[1] - bar_y_position) <= y_tolerance

        # The ball is considered to be 'on' the bar if both conditions are True
        return is_within_x_range and is_close_to_y_position
        
# Define the policy model using TensorFlow
class PolicyModel(tf.keras.Model):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense3 = layers.Dense(1, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# Create the environment and the model
env = BouncingBallEnv()
model = PolicyModel()

# Training loop
optimizer = tf.optimizers.Adam(learning_rate=0.01)
for episode in range(1000):
    with tf.GradientTape() as tape:
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action = model(state_tensor)
            state, reward, done, _ = env.step(action.numpy())
            #episode_reward += reward
            episode_reward += tf.cast(reward, tf.float32)  # Ensure reward is a TensorFlow tensor

        # Compute the gradient and update the model
        grads = tape.gradient(episode_reward, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Episode: {episode} Reward: {episode_reward}")
