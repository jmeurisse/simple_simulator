# import mujoco
# import numpy as np
# import PIL.Image

# width=1920
# height=1080
# gl = mujoco.GLContext(1920, 1080)
# gl.make_current()

# model = mujoco.MjModel.from_xml_path("bouncing_ball.xml")
# data = mujoco.MjData(model)
# mujoco.mj_forward(model, data)

# scene = mujoco.MjvScene(model, maxgeom=10000)
# cam = mujoco.MjvCamera()
# mujoco.mjv_defaultCamera(cam)
# cam.lookat[0] = 0  # x-position of the point to look at (origin)
# cam.lookat[1] = 0  # y-position of the point to look at (origin)
# cam.lookat[2] = 1  # z-position of the point to look at (origin)
# cam.distance = 4 # Distance from the point to look at
# cam.azimuth = 90 # Rotation around the vertical axis, in degrees
# cam.elevation = -30 # Angle above the horizon, in degrees

# mujoco.mjv_updateScene(
#     model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
#     cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)

# context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
# mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, context)

# viewport = mujoco.MjrRect(0, 0, 640, 480)
# mujoco.mjr_render(viewport, scene, context)

# upside_down_image = np.empty((480, 640, 3), dtype=np.uint8)
# mujoco.mjr_readPixels(upside_down_image, None, viewport, context)
# PIL.Image.fromarray(np.flipud(upside_down_image)).save("img.png")

import mujoco
import numpy as np
import PIL.Image

width = 624
height = 420
gl = mujoco.GLContext(width, height)
gl.make_current()

model = mujoco.MjModel.from_xml_path("bouncing_ball.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

scene = mujoco.MjvScene(model, maxgeom=10000)
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
    cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)

context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, context)

# Set viewport dimensions to match the resolution
viewport = mujoco.MjrRect(0, 0, width, height)

# Ensure the NumPy array matches the new viewport dimensions
upside_down_image = np.empty((height, width, 3), dtype=np.uint8)

mujoco.mjr_render(viewport, scene, context)
mujoco.mjr_readPixels(upside_down_image, None, viewport, context)

# Flip the image and save
PIL.Image.fromarray(np.flipud(upside_down_image)).save("img.png")
