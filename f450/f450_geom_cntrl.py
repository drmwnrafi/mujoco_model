import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
from controller import GeometricControl
from traj_gen.differential_flatness import TrajectoryOptimizer
from mouse_callbacks import *
from scipy.interpolate import interp1d
        
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if (not button_left) and (not button_middle) and (not button_right):
        return

    width, height = glfw.get_window_size(window)

    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)
    
def render_waypoints(scene, waypoints):
    for waypoint in waypoints:
        # Render the waypoint as a box
        mj.mjv_initGeom(
            geom=scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_BOX,
            size=0.2 * np.array([1, 1, 1], dtype=np.float64),
            pos=np.array(waypoint[:3], dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 0.5], dtype=np.float32)
        )
        scene.ngeom += 1

        # Compute direction vector based on yaw (4th element in waypoint)
        yaw = waypoint[3]  # Assuming yaw in radians
        direction = np.array([np.cos(yaw), np.sin(yaw), 0], dtype=np.float64)
        arrow_start = np.array(waypoint[:3], dtype=np.float64)
        arrow_end = arrow_start + 0.5 * direction  # Scale direction length

        # Render the direction as an arrow (capsule)
        mj.mjv_initGeom(
            geom=scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.05, 0, 0], dtype=np.float64),  # Capsule radius
            pos=(arrow_start + arrow_end) / 2,
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 0, 1, 1], dtype=np.float32)  # Blue arrow
        )
        scene.ngeom += 1
        
def render_trajectory(scene, trajectory_points):
    for i in range(len(trajectory_points) - 1):
        point1 = trajectory_points[i]
        point2 = trajectory_points[i + 1]

        direction = point2 - point1
        length = np.linalg.norm(direction)

        if length > 0:
            direction /= length
            radius = 0.05
            geom = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(geom, 
                            mj.mjtGeom.mjGEOM_CAPSULE, 
                            np.zeros(3), 
                            np.zeros(3), 
                            np.zeros(9), 
                            np.array([0.0, 1.0, 1.0, 0.5], 
                                     dtype=np.float32))
            mj.mjv_connector(geom, 
                             mj.mjtGeom.mjGEOM_CAPSULE, 
                             radius, 
                             point1, 
                             point2)
            scene.ngeom += 1

xml_path = "f450/scene_low_poly.xml"
# xml_path = "scene.xml"
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
cam.type = mj.mjtCamera.mjCAMERA_FREE
opt = mj.MjvOption()

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

glfw.init()
window = glfw.create_window(900, 900, "Geometric Control Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultFreeCamera(model, cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_100.value)

glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = 40
cam.elevation = -30
cam.distance = 40
cam.lookat = np.array([0.0, 0.0, 0.0])

param = {
    "J": np.zeros((3, 3)),  
    "m": 1.325,  
    "g": 9.81    
}

H = np.zeros((model.nv, model.nv))
mj.mj_fullM(model, H, data.qM)
param["J"] = H[-3:, -3:]
print(param["J"])
# kx, kv, kR, komega = np.array([25, 25, 170]), np.array([8, 8, 20]), 30, 2.5  # control gains
# kx, kv, kR, komega = 10, 2, 25, 5  # control gains
kx, kv, kR, komega = 10, 5, 25, 5  # control gains

waypoints = np.array([
    [0, 0, 0.35, 0],
    [0, 5, 5, 0],
    [5, 0, 5, 0],
    [0, -5, 5, 0],
    [-5, 0, 5, 0],
    [0, 5, 10, 0],
])
n_coeffs = [8, 8, 8, 4]  
derivatives = [4, 4, 4, 2] 
times = [0, 4, 8, 12, 16, 22]

optimizer = TrajectoryOptimizer(n_coeffs, derivatives, times)
states, coeff = optimizer.generate_trajectory(waypoints, num_points=200)

controller = GeometricControl(kx, kv, kR, komega)

x_traj = states[0][0,:]
y_traj = states[1][0,:]
z_traj = states[2][0,:]

trajectory_points = np.vstack([x_traj, y_traj, z_traj]).T

kf = 1
km = 0.0201
dx = 0.79450535
dy = 0.7945588
# dx = 0.14
# dy = 0.18
# c = 0.0201
N = np.array([[kf, kf, kf, kf],
              [-kf*dy, 0, 0, -kf*dy],
              [0, 0, -kf*dx, -kf*dx],
              [-km, km, -km, km]])
N_inv = np.linalg.inv(N)

motor_names = ["motor1", "motor2", "motor3", "motor4"]
motor_ids = [model.actuator(name).id for name in motor_names]


while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0 / 60.0):
        qpos = data.qpos
        qvel = data.qvel

        position = qpos[:3]
        quaternion = qpos[3:7]
        velocity = qvel[:3]
        angular_velocity = qvel[3:6]
        
        x = np.concatenate((position, quaternion, velocity, angular_velocity))

        t = data.time
        Fz, M = controller.controller(t, x, states, times, param)

        motor_inputs = controller.actuator_allocation(Fz, M, N_inv)

        for motor_id, force in zip(motor_ids, motor_inputs):
            data.ctrl[motor_id] = force  

        mj.mj_step(model, data)

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    render_waypoints(scene, waypoints)
    render_trajectory(scene, trajectory_points)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
