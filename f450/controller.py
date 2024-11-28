import numpy as np
from scipy.spatial.transform import Rotation as R

class GeometricControl:
    @staticmethod
    def hat_map(v):
        """Hat map for a vector."""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    @staticmethod
    def vee_map(skew):
        """Vee map for a skew-symmetric matrix."""
        return 0.5 * np.array([skew[2, 1] - skew[1, 2],
                               skew[0, 2] - skew[2, 0],
                               skew[1, 0] - skew[0, 1]])

    @staticmethod
    def dir_cosine(q):
        """Compute the direction cosine matrix from a quaternion."""
        C_B_I = np.zeros((3, 3))
        C_B_I[0, 0] = 1 - 2 * (q[2]**2 + q[3]**2)
        C_B_I[0, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
        C_B_I[0, 2] = 2 * (q[1] * q[3] - q[0] * q[2])
        C_B_I[1, 0] = 2 * (q[1] * q[2] - q[0] * q[3])
        C_B_I[1, 1] = 1 - 2 * (q[1]**2 + q[3]**2)
        C_B_I[1, 2] = 2 * (q[2] * q[3] + q[0] * q[1])
        C_B_I[2, 0] = 2 * (q[1] * q[3] + q[0] * q[2])
        C_B_I[2, 1] = 2 * (q[2] * q[3] - q[0] * q[1])
        C_B_I[2, 2] = 1 - 2 * (q[1]**2 + q[2]**2)
        return C_B_I.T

    @staticmethod
    def controller(t, x, kR, komega, kx, kv, param):
        """Geometric control."""
        g = param["g"]
        m = param["m"]
        J = param["J"]

        # Extract state variables
        rx, ry, rz = x[:3]
        quat = x[3:7]
        vx, vy, vz = x[7:10]
        omega = x[10:13]

        # Rotation matrix
        R = GeometricControl.dir_cosine(quat)

        # Desired position
        frequency = 2 * np.pi / 5
        radius = 8
        zdes = 10.0
        zdes_dot = 0.0
        zdes_ddot = 0.0

        xdes = radius * np.cos(frequency * t)
        xdes_dot = -radius * frequency * np.sin(frequency * t)
        xdes_ddot = -radius * frequency**2 * np.cos(frequency * t)
        ydes = radius * np.sin(frequency * t)
        ydes_dot = radius * frequency * np.cos(frequency * t)
        ydes_ddot = -radius * frequency**2 * np.sin(frequency * t)

        psi_des = np.deg2rad(0)

        # Errors
        ex = np.array([rx, ry, rz]) - np.array([xdes, ydes, zdes])
        ev = np.array([vx, vy, vz]) - np.array([xdes_dot, ydes_dot, zdes_dot])
        
        # print(f"{np.array([rx, ry, rz])} == {np.array([xdes, ydes, zdes])}")
        # print(f"{np.array([vx, vy, vz])} == {np.array([xdes_dot, ydes_dot, zdes_dot])}")
        
        # Total thrust
        xddot_des = np.array([xdes_ddot, ydes_ddot, zdes_ddot])
        Fd = -kx * ex - kv * ev + m * g * np.array([0, 0, 1]) + m * xddot_des
        Fz = np.dot(Fd, R @ np.array([0, 0, 1]))

        # Desired attitudes
        if np.linalg.norm(Fd) < 1e-8:
            raise ValueError("Fd is too small")
        b3d = Fd / np.linalg.norm(Fd)
        b1d = np.array([np.cos(psi_des), np.sin(psi_des), 0])
        b2d = np.cross(b3d, b1d)
        b2d = b2d / np.linalg.norm(b2d)
        Rd = np.zeros((3, 3))
        Rd[:, 0] = np.cross(b2d, b3d)
        Rd[:, 1] = b2d
        Rd[:, 2] = b3d
        omega_des = np.zeros(3)
        omega_dot_des = np.zeros(3)

        # Errors
        eR = 0.5 * GeometricControl.vee_map(Rd.T @ R - R.T @ Rd)
        e_omega = omega - R.T @ Rd @ omega_des

        # Moments
        M = ((-kR * eR - komega * e_omega) +
             np.cross(omega, J @ omega) -
             J @ (GeometricControl.hat_map(omega) @ R.T @ Rd @ omega_des - R.T @ Rd @ omega_dot_des))

        return Fz, M

    @staticmethod
    def actuator_allocation(Fz, M, N_inv):
        """Compute motor inputs from force and moment."""
        return N_inv @ np.array([Fz, *M])
