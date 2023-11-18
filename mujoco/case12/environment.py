import os, sys
expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'."+
             f" \nPlease activate the correct environment and try again.\nconda activate '{expected_env}'")

xml = """
<mujoco model="bouncer">
  <compiler inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
        <light pos="0 3 3" dir="0 -1 -1" diffuse="1.5 1.5 1.5"/>
        <light pos="0 0 4" dir="0 0 -1" diffuse="2 2 2"/>
        <light pos="-2 3 3" dir="1 -1 -1" diffuse="1 1 1"/>
        <light pos="2 3 3" dir="-1 -1 -1" diffuse="1 1 1"/>
        <!-- Left Wall -->
        <body name="left_wall" pos="-2 0 1"> <!-- Adjust the position as needed -->
            <geom name="left_wall_geom" type="box" size="0.1 0.1 2" rgba="0.8 0.9 0.8 1"/>
        </body>

        <!-- Right Wall -->
        <body name="right_wall" pos="2 0 1"> <!-- Adjust the position as needed -->
            <geom name="right_wall_geom" type="box" size="0.1 0.1 2" rgba="0.8 0.9 0.8 1"/>
        </body>

        <!-- Floor -->
        <body name="floor" pos="0 0 0">
            <geom name="floor_geom" type="plane" size="2 1 0.1" rgba="0.8 0.9 0.8 1"/>
        </body>

        <body name="bar" pos="0 0 1">
            <geom name="bar_geom" type="box" size="0.5 0.1 0.1" rgba="0.8 0.1 0.1 1"/>
            <joint name="bar_joint" type="slide" axis="1 0 0"/>
        </body>
        <body name="ball" pos="0 0 1.5">
            <geom name="ball_geom" type="ellipsoid" size="0.17 0.11 0.11"
                  density="100" rgba="0.1 0.1 0.8 1" solref="-1000 0"/>
            <joint name="ball_slide_x" type="slide" axis="1 0 0"/>
            <joint name="ball_slide_z" type="slide" axis="0 0 1"/>
            <joint name="ball_hinge_y" type="hinge" axis="0 1 0"/>
        </body>
    </worldbody>
    <actuator>
        <motor joint="bar_joint" ctrlrange="-500 500" ctrllimited="true"/>
    </actuator>
</mujoco>
"""
import os, sys
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
        self.model = mujoco.MjModel.from_xml_string(xml)
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
        self.bar_center_x_pos_table = []
        self.bar_center_x_vel_table = []
        self.ball_center_x_pos_table = []
        self.ball_center_z_pos_table = []
        self.rewards_direction = []
        self.rewards_wall = []
        self.rewards_range_pos = []
        self.rewards_range_vel = []

    def step(self, logits):
        epsilon=0.1
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
        reward = self.compute_reward(xpos, xvel)

        # Define done condition
        ball_z_position = self.data.geom_xpos[self.ball_geom_id][2] # z ball position
        floor_z_position = 0.1  # between the height of the bar and the height of the floor
        done = ball_z_position <= floor_z_position

        return state, reward, done, {}

    def reset(self, random_flag=True, init_angle_value=0):
        self.bar_center_x_pos_table = []
        self.bar_center_x_vel_table = []
        self.ball_center_x_pos_table = []
        self.ball_center_z_pos_table = []
        self.rewards_direction = []
        self.rewards_wall = []
        self.rewards_range_pos = []
        self.rewards_range_vel = []

        # inital positions
        self.bar_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'bar_geom')
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')
        self.left_wall_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'left_wall_geom')
        self.right_wall_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'right_wall_geom')
        self.init_bar_xpos = self.data.geom_xpos[self.bar_geom_id][0]
        self.init_bar_zpos = self.data.geom_xpos[self.bar_geom_id][2]
        self.init_ball_xpos = self.data.geom_xpos[self.ball_geom_id][0]
        self.init_ball_zpos = self.data.geom_xpos[self.ball_geom_id][2]
        self.init_left_wall_xpos = self.data.geom_xpos[self.left_wall_id][0]
        self.init_right_wall_xpos = self.data.geom_xpos[self.right_wall_id][0]

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
        # give initial condition aligned
        init_speed = -0.5 # m/s
        state[7] = init_speed # bar x velocity
        self.data.qvel[0] = init_speed # bar x velocity
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

    def positional_reward(self, ball_x, bar_center_x, max_distance):
        distance = abs(ball_x - bar_center_x)
        reward = np.exp(-distance**2 / (2 * max_distance**2))
        return reward

    def stability_reward(self, ball_velocity, max_velocity):
        reward = 1.0 - min(abs(ball_velocity) / max_velocity, 1.0)
        return reward

    def compute_reward(self, xpos, xvel):
        bar_center_z_pos = self.data.geom_xpos[self.bar_geom_id][2]
        bar_center_x_pos = self.data.geom_xpos[self.bar_geom_id][0]
        bar_center_x_vel = self.data.qvel[0]

        ball_center_x_pos = self.data.geom_xpos[self.ball_geom_id][0]
        ball_center_z_pos = self.data.geom_xpos[self.ball_geom_id][2]
        ball_center_x_vel = self.data.qvel[1]

        bar_x_min = -0.25 + bar_center_x_pos
        bar_x_max = 0.25 + bar_center_x_pos
        is_within_x_range = bar_x_min <= ball_center_x_pos <= bar_x_max

        bar_z_position = 1
        bar_z_tol = 0.1
        is_close_to_the_bar = bar_z_position <= ball_center_z_pos # <= bar_z_position + bar_z_tol

        wall_size = 0.1/2
        left_wall_x_pos =  self.data.geom_xpos[self.left_wall_id][0] + wall_size
        right_wall_x_pos =  self.data.geom_xpos[self.right_wall_id][0] - wall_size
        wall_tol = 0.6
        is_close_to_the_wall = left_wall_x_pos + wall_tol >= bar_center_x_pos or bar_center_x_pos >= right_wall_x_pos - wall_tol

        self.bar_center_x_pos_table.append(bar_center_x_pos)
        self.bar_center_x_vel_table.append(bar_center_x_vel)
        self.ball_center_x_pos_table.append(ball_center_x_pos)
        self.ball_center_z_pos_table.append(ball_center_z_pos)
        
        # initial
        total_reward=0
        
        # survival reward
        total_reward += 0.01

        # # desired direction
        # desired_direction = np.sign(ball_center_x_pos - bar_center_x_pos)
        # direction_alignment = np.sign(bar_center_x_vel) == desired_direction
        # direction_reward = 0.1 if direction_alignment else 0
        # total_reward += direction_reward
        # self.rewards_direction.append(direction_reward)

        # wall reward
        wall_reward = 0 if is_close_to_the_wall else 0.5
        total_reward += wall_reward
        self.rewards_wall.append(wall_reward)

        # if close to the bar and within x range
        pos_reward = -0.1
        stab_reward = 0
        if is_within_x_range and is_close_to_the_bar:
            max_distance = 1 * 0.25 # bar length/2 = 0.25 m
            max_velocity = 5 # m/s
            pos_reward = self.positional_reward(ball_center_x_pos, bar_center_x_pos, max_distance)
            stab_reward = 0 #self.stability_reward(ball_center_x_vel, max_velocity)
        total_reward += pos_reward + stab_reward
        self.rewards_range_pos.append(pos_reward)
        self.rewards_range_vel.append(stab_reward)

        return total_reward
