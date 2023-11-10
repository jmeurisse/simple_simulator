import os, sys
expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'."+
             f" \nPlease activate the correct environment and try again.\nconda activate '{expected_env}'")

import gym, mujoco, imageio
import numpy as np
from mujoco import _render
from mujoco import _enums

# Define the MuJoCo Gym environment
class BouncingBallEnv(gym.Env):
    def __init__(self):
    
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32) # same size than state

        # Create an OpenGL context
        self.gl_context = mujoco.GLContext(1920, 1080)
        self.gl_context.make_current()

        # Load the model from an XML file
        self.model = mujoco.MjModel.from_xml_path("bouncing_ball.xml")
        self.data = mujoco.MjData(self.model)

        # Render        
        self.scn = mujoco.MjvScene(self.model, maxgeom=1000)
        self.cam = mujoco.MjvCamera()  # Initialize the camera
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.width = 1920
        self.height = 1088
        self.frames = []
        

    def step(self, action):
        self.data.ctrl[0] = action[0][0]
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Get the state
        xpos = self.data.qpos[:]
        xvel = self.data.qvel[:]
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
        state[3] = np.deg2rad(45)
        return state

    def render(self, mode):
        # Ensure the OpenGL context is current
        self.gl_context.make_current()

        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)

        if mode == 'human':
            print(self.data)
        elif mode == 'image':
            # self.renderer.update_scene(self.data)
            image = self.renderer.render()
            print("Create envir.png")
            imageio.imwrite('envir.png', image)
        elif mode == 'video':
            # Allocate an array to store the pixels
            pixels = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Render the scene into the pixel buffer
            mujoco.mjr_render(mujoco.MjrRect(0, 0, self.width, self.height), self.scn, self.ctx)
            
            # Read the pixels from the buffer
            mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, self.width, self.height), self.ctx)
            
            # Add the frame to the list
            self.frames.append(pixels)
        else:
            raise NotImplementedError(f"Mode '{mode}' not supported.")
        
    def save_video(self, video_name, framerate=60):
        print ("Video saved as "+video_name)
        with imageio.get_writer(video_name, fps=framerate) as writer:
            for frame in self.frames:
                writer.append_data(frame)

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
        
    def __del__(self):
        # Clean up the OpenGL context when the environment is deleted
        self.ctx.free()
        del self.gl_context