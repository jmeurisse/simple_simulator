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
import random

# Define the MuJoCo Gym environment
class BouncingBallEnv(gym.Env):
    def __init__(self):
    
        # Define action and observation space
        self.max_force_ctrl = 500
        self.action_space = gym.spaces.Box(low=-self.max_force_ctrl, high=self.max_force_ctrl, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32) # same size than state

        # Create an OpenGL context
        self.width = 640
        self.height = 480
        self.gl_context = mujoco.GLContext(self.width, self.height)
        self.gl_context.make_current()

        # Load the model from an XML file
        self.model = mujoco.MjModel.from_xml_path("bouncing_ball.xml")
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Rendering
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[0] = 0  # x-position of the point to look at (origin)
        self.cam.lookat[1] = 0  # y-position of the point to look at (origin)
        self.cam.lookat[2] = 1  # z-position of the point to look at (origin)
        self.cam.distance = 4 # Distance from the point to look at
        self.cam.azimuth = 90 # Rotation around the vertical axis, in degrees
        self.cam.elevation = -30 # Angle above the horizon, in degrees

        mujoco.mjv_updateScene(
        self.model, self.data, mujoco.MjvOption(), mujoco.MjvPerturb(),
        mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL.value, self.scn)

        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.ctx)

        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)     

        self.frames = []
        
    def step(self, logits):
        #self.data.ctrl[0] = logits[0][0]*self.max_force_ctrl/10 
        epsilon=0.2
        ratio=0.1
        if np.random.rand() < epsilon:
            # Perform random action - ensure it's within your action space bounds
            self.data.ctrl[0] = np.random.uniform(low=-self.max_force_ctrl*ratio, high=self.max_force_ctrl*ratio)
        else:
            # Perform the action suggested by the model
            self.data.ctrl[0] = logits[0][0]*self.max_force_ctrl*ratio
      
        mujoco.mj_forward(self.model, self.data)

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
        floor_z_position = -1.2  # between the height of the bar and the height of the floor
        done = ball_z_position <= floor_z_position

        return state, reward, done, {}

    def reset(self, random_flag=True, init_angle_value=0):
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)
        xpos = self.data.qpos[:]
        xvel = self.data.qvel[:]
        state = np.concatenate([xpos, xvel])
        if random_flag:
            self.random_integer = random.randint(0, 360)
            init_angle = self.random_integer # degrees
        else:
            init_angle = init_angle_value
        state[3] = np.deg2rad(init_angle)
        self.data.qpos[3] = np.deg2rad(init_angle)
        return state

    def render(self, mode):
        # Ensure the OpenGL context is current
        self.gl_context.make_current()
        mujoco.mj_forward(self.model, self.data)

        mujoco.mjv_updateScene(
        self.model, self.data, mujoco.MjvOption(), mujoco.MjvPerturb(),
        self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scn)

        if mode == 'human':
            print(self.data)
        elif mode == 'image':
            print("Create envir.png")
            
            upside_down_image = np.empty((self.height, self.width, 3), dtype=np.uint8)
            mujoco.mjr_render(self.viewport, self.scn, self.ctx)
            mujoco.mjr_readPixels(upside_down_image, None, self.viewport, self.ctx)
            right_side_up_image = np.flipud(upside_down_image)
            imageio.imwrite('envir.png', right_side_up_image)

        elif mode == 'video':
            # Allocate an array to store the pixels
            pixels = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
            # Render the scene into the pixel buffer
            mujoco.mjr_render(self.viewport, self.scn, self.ctx)
            
            # Read the pixels from the buffer
            mujoco.mjr_readPixels(pixels, None, self.viewport, self.ctx)
            
            # Add the frame to the list
            self.frames.append(np.flipud(pixels))
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
        
