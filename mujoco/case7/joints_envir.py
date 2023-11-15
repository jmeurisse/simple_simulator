import mujoco
import imageio
import os, sys
import numpy as np

expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'. Please activate the correct environment and try again.")

width = 640
height = 480
gl_context = mujoco.GLContext(width, height)
gl_context.make_current()
model = mujoco.MjModel.from_xml_path('bouncing_ball.xml')
data = mujoco.MjData(model)
scn = mujoco.MjvScene(model, maxgeom=10000)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[0] = 0  # x-position of the point to look at (origin)
cam.lookat[1] = 0  # y-position of the point to look at (origin)
cam.lookat[2] = 1  # z-position of the point to look at (origin)
cam.distance = 4 # Distance from the point to look at
cam.azimuth = 90 # Rotation around the vertical axis, in degrees
cam.elevation = -30 # Angle above the horizon, in degrees

mujoco.mjv_updateScene(
model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL.value, scn)

ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, ctx)

viewport = mujoco.MjrRect(0, 0, width, height)
mujoco.mjr_render(viewport, scn, ctx)     

data.qpos[3]=np.deg2rad(45)
mujoco.mj_forward(model, data)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

gl_context.make_current()
mujoco.mj_forward(model, data)
mujoco.mjv_updateScene(
model, data, scene_option, mujoco.MjvPerturb(),
cam, mujoco.mjtCatBit.mjCAT_ALL.value, scn)
print("Create joints_envir.png")
upside_down_image = np.empty((height, width, 3), dtype=np.uint8)
mujoco.mjr_render(viewport, scn, ctx)
mujoco.mjr_readPixels(upside_down_image, None, viewport, ctx)
right_side_up_image = np.flipud(upside_down_image)
imageio.imwrite('joints_envir.png', right_side_up_image)
