import tensorflow as tf
from gym_envir import BouncingBallEnv
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
env = BouncingBallEnv()
state = env.reset()
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
print(f"Rewards sum: {np.sum(episode_rewards)}")
