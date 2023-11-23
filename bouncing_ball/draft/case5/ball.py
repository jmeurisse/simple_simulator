import mujoco
import numpy as np
import PIL.Image

ratio=[1,1.2,1.3,1.4]
init_w=640
init_h=480

# 832x624 works

max_w=init_w*max(ratio)
max_h=init_h*max(ratio)

gl = mujoco.GLContext(init_w, init_h)
gl.make_current()

XML = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

for i in ratio:
    w = int(init_w * i)
    h = int(init_h * i)

    # Update data for the new loop iteration
    mujoco.mj_forward(model, data)

    # Create and update scene for each iteration
    scene = mujoco.MjvScene(model, maxgeom=10000)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
                           mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL.value, scene)

    # Create context and set buffer
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, context)

    # Update the viewport for each iteration
    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjr_render(viewport, scene, context)

    # Update the image buffer size
    upside_down_image = np.empty((h, w, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(upside_down_image, None, viewport, context)
    PIL.Image.fromarray(np.flipud(upside_down_image)).save("ball_" + str(w) + "x" + str(h) + ".png")