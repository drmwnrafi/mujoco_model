from simple_pid import PID
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

# Initialize Mujoco
xml_path = "f450/scene.xml"

# Initialize Mujoco model and data
model = mj.MjModel.from_xml_path(xml_path) 
data = mj.MjData(model)               
cam = mj.MjvCamera()                        
opt = mj.MjvOption()                       

# GLFW setup
glfw.init()
window = glfw.create_window(500, 500, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Camera settings
mj.mjv_defaultFreeCamera(model, cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.azimuth = 40
cam.elevation = -30
cam.distance = 30
cam.lookat = np.array([0.0, 0.0, 0.0])

# PID Controllers
PID_x        = PID(5.2, 2.08, 0.06)
PID_y        = PID(5.2, 2.08, 0.06)

PID_throttle = PID(5.0, 3.3, 0.02)   # setpoint from desired height
PID_roll     = PID(4.7, 1.3, 0.0)
PID_pitch    = PID(4.7, 1.3, 0.0)
PID_yaw      = PID(1.7, 1.2, 0.0) # ki set to 0

PID_AccX     = PID(1.4, 1.5, 0.0)  # setpoint from PID_roll output
PID_AccY     = PID(1.4, 1.5, 0.0)  # setpoint from PID_pitch output 
PID_AccZ     = PID(1.6, 1.5, 0.0)  # setpoint from PID_yaw output

# Motor names
motor_names = ["motor1", "motor2", "motor3", "motor4"]
motor_ids = [model.actuator(name).id for name in motor_names]

# Get the joint ID for the free joint
joint_id =  model.joint("CoG").id

# The starting index of this joint's qpos and qvel in the arrays
qpos_start = model.jnt_qposadr[joint_id]
qvel_start = model.jnt_dofadr[joint_id]

# Target position
target = [0, 0, 5]

# Update PID setpoints
PID_x.setpoint = target[0]
PID_y.setpoint = target[1]
PID_throttle.setpoint = target[2]

normalize_angle = lambda angle: (angle + np.pi) % (2 * np.pi) - np.pi

# Simulation Loop
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0 / 60.0):
        # Read state (position, velocity, orientation)
        qpos = data.qpos[qpos_start : qpos_start + 7]  # [x, y, z, qw, qx, qy, qz]
        qvel = data.qvel[qvel_start : qvel_start + 6]  # [vx, vy, vz, wx, wy, wz]
        
        position = qpos[0:3]
        quaternion = qpos[3:7]
        rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
        
        desired_pitch = PID_x(position[0])
        desired_roll  = PID_y(position[1])
        thrust        = PID_throttle(position[2])
        
        PID_roll.setpoint =  normalize_angle(desired_roll + np.pi / 2)
        PID_pitch.setpoint = desired_pitch
        # PID_roll.setpoint =  np.pi / 2
        # PID_pitch.setpoint = 0
        
        lin_vel = qvel[0:3]
        ang_vel = qvel[3:6]
        
        desired_accx = PID_roll(roll)
        desired_accy = PID_pitch(pitch)
        desired_accz = PID_yaw(yaw)
        
        PID_AccX.setpoint = desired_accx
        PID_AccY.setpoint = desired_accy
        PID_AccZ.setpoint = desired_accz
        
        output_acc_x = PID_AccX(ang_vel[0])
        output_acc_y = PID_AccY(ang_vel[1])
        output_acc_z = PID_AccZ(ang_vel[2])
        
        # Compute PID outputs
        roll_output     = PID_AccX(roll)
        pitch_output    = PID_AccY(pitch)
        yaw_output      = PID_AccZ(yaw)

        # Map PID outputs to motor thrusts
        motor_forces = np.zeros(4)
        motor_forces[0] = thrust + roll_output - yaw_output  # Motor 1
        motor_forces[1] = thrust - roll_output + yaw_output  # Motor 2
        motor_forces[2] = thrust - pitch_output - yaw_output  # Motor 3
        motor_forces[3] = thrust + pitch_output + yaw_output  # Motor 4
        
        print(f"X, Y, Z = {position}")

        # Apply forces to actuators
        for motor_id, force in zip(motor_ids, motor_forces):
            data.ctrl[motor_id] = max(0, min(force, 15))  # Clamp force for safety

        mj.mj_step(model, data)

    # Rendering
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
