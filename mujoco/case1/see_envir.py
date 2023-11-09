import mujoco
import imageio
import os, sys

expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'. Please activate the correct environment and try again.")

model = mujoco.MjModel.from_xml_path('bouncing_ball.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)

#id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')
#xpos = data.qpos[id]
#xvel = data.qvel[id]

joint_dofs = {
    mujoco.mjtJoint.mjJNT_HINGE: 1,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_BALL: 3,
    mujoco.mjtJoint.mjJNT_FREE: 6,
}

print('Total number of DoFs in the model:', model.nv)
print('Total number of joints in the model:', model.njnt)
for i in range(0,model.njnt):
    print("joint "+str(i)+", name=",model.joint(i).name, ", type=", mujoco.mjtJoint(model.jnt_type[i]).name,", DOF=", joint_dofs.get(mujoco.mjtJoint(model.jnt_type[i]), 0))
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)

print('Total number of actuators in the model:', model.nu) # => len(data.ctrl)=1

print('Total number of geometries in the model:', model.ngeom)
for i in range(0,model.ngeom):
    print("geom ",i,model.geom(i).name)
print('Geom positions:', data.geom_xpos)

print('Total number of bodies in the model:', model.nbody)
for i in range(0,model.nbody):
    print("body ",i,model.body(i).name)

# get id
# id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'slide_joint')

# positions of DOF
#data.qpos[0]=-0.5     # SLIDE translation X
#data.qpos[1]=-0.3     # SPHERE translation X   
#data.qpos[2]=0.01     # SPHERE translation Y
#data.qpos[3]=1.3      # SPHERE translation Z
#data.qpos[4]=sin(45)  # SPHERE rotation around X

# AXIS
# Z
# ^
# |
# x --> X
# Y
# BALL + SLIDE
#   O
# ___

# Actuator
# data.ctrl[0]=-5 # Force in Newton on slide X axis

# Force or Torque Actuators (default for <motor> tags): The unit of control is typically force in Newtons (N) or torque in Newton-meters (Nm).
# Position Actuators: If the actuator is a position servo, then the control signal would be in the units of position (e.g., meters for translational joints, radians for rotational joints).
# Velocity Actuators: If the actuator is controlling velocity, then the control input would be in units of velocity (e.g., meters per second for translational joints, radians per second for rotational joints).


# update data
mujoco.mj_forward(model, data)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Image
renderer.update_scene(data)
image = renderer.render()
print("Create envir.png")
imageio.imwrite('envir.png', image)

# video
duration = 10  # (seconds)
framerate = 60  # (Hz)

frames = []
# mujoco.mj_resetData(model, data)
while data.time < duration:
  mujoco.mj_step(model, data)
  # print(data.qpos[0])
  if len(frames) < data.time * framerate:
    renderer.update_scene(data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)


# Simulate and display video.
video_name="mujoco_video.mp4"
with imageio.get_writer(video_name, fps=framerate) as writer:
    for frame in frames:
        writer.append_data(frame)

print("Video saved as ",video_name)
