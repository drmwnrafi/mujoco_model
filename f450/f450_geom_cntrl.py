import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco as mj
from mujoco.glfw import glfw
import time
from controller import GeometricControl
from mouse_callbacks import *

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

xml_path = "scene_low_poly.xml"
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
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = -40
cam.elevation = -40
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
kR, komega, kx, kv = 42, 25, 10, 8  # control gains

dx = 0.79450535
dy = 0.7945588
c = 0.0201
N = np.array([[1, 1, 1, 1],
              [-dy, dy, dy, -dy],
              [dx, dx, -dx, -dx],
              [-c, c, -c, c]])
N_inv = np.linalg.inv(N)

motor_names = ["motor1", "motor2", "motor3", "motor4"]
motor_ids = [model.actuator(name).id for name in motor_names]

while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0 / 60.0):
        # State extraction
        # qpos = data.qpos[qpos_start: qpos_start + 7]
        # qvel = data.qvel[qvel_start: qvel_start + 6]
        
        qpos = data.qpos
        qvel = data.qvel

        position = qpos[:3]
        quaternion = qpos[3:7]
        velocity = qvel[:3]
        angular_velocity = qvel[3:6]

        x = np.concatenate((position, quaternion, velocity, angular_velocity))

        t = data.time
        Fz, M = GeometricControl.controller(t, x, kR, komega, kx, kv, param)

        motor_inputs = GeometricControl.actuator_allocation(Fz, M, N_inv)

        for motor_id, force in zip(motor_ids, motor_inputs):
            data.ctrl[motor_id] = force  

        mj.mj_step(model, data)

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
