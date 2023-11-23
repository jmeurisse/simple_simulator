import mujoco
import imageio
import os, sys

expected_env = 'ffmpeg'
current_env = os.environ.get('CONDA_DEFAULT_ENV')
if current_env != expected_env:
    sys.exit(f"Error: This script requires the Conda environment '{expected_env}', but the current environment is '{current_env}'. Please activate the correct environment and try again.")

model = mujoco.MjModel.from_xml_path('robot_arm.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)

# # print(model.ngeom)
# id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'arm_flex')
# # print(model.joint(id).name)
# data.qpos[0] = 0.3
# data.qpos[id] = 0.7
# data.qvel[id] = 10

# List of target joint positions
target_joint_positions = [
    (0, 0.5, 0.5, 0),  # Target positions for each joint
    (0.1, -0.5, -0.5, 0.1),
]

# Helper function to compute control signal
def proportional_control(target, current):
    kp = 0.1  # Proportional gain, this needs to be tuned
    return kp * (target - current)

# Start the simulation
for target_position in target_joint_positions:
    while True:
        # Apply control to each joint
        for i, joint_name in enumerate(['base_rotate', 'arm_flex', 'elbow_flex', 'wrist_flex']):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            # Calculate control signal
            control_signal = proportional_control(target_position[i], data.qpos[joint_id])
            data.ctrl[joint_id] = control_signal

        # Step the simulation
        mujoco.mj_step(model, data)

        # Check if we are close enough to the target position for all joints
        if all(abs(data.qpos[joint_id] - target_position[i]) < 0.01 for i, joint_name in enumerate(['base_rotate', 'arm_flex', 'elbow_flex', 'wrist_flex'])):
            break  # Move to the next target position

        # Update the scene for visualization
        renderer.render()

# Log initial joint positions
print(f"Initial joint positions: {data.qpos}")

# # Set initial joint positions (angles)
# data.joint('arm_flex').qpos = 0.785
# data.joint('arm_flex').qvel = 1



# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

frames = []
# mujoco.mj_resetData(model, data)
while data.time < duration:
  mujoco.mj_step(model, data)
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
