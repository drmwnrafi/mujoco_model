import numpy as np
from scipy.spatial.transform import Rotation as R

class GeometricControl:
    def __init__(self, kx, kv, kR, komega):
        """Initialize control gains."""
        self.kx = kx
        self.kv = kv
        self.kR = kR
        self.komega = komega

    def hat_map(self, v):
        """Hat map for a vector."""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def vee_map(self, skew):
        """Vee map for a skew-symmetric matrix."""
        return 0.5 * np.array([skew[2, 1] - skew[1, 2],
                               skew[0, 2] - skew[2, 0],
                               skew[1, 0] - skew[0, 1]])

    def controller(self, t, x, x_des, t_des, param):
        """Geometric control."""
        g = param["g"]
        m = param["m"]
        J = param["J"]

        # Extract state variables
        rx, ry, rz = x[:3]
        quat = x[3:7]
        vx, vy, vz = x[7:10]
        omega = x[10:13]

        # Convert quaternion to rotation matrix using scipy
        rotation = R.from_quat(quat, scalar_first=True).as_matrix()

        N = len(x_des[0][0])  # Number of interpolated points
        time_idx = min(int(t * (N / t_des[-1])), N - 1)
        xdes, xdes_dot, xdes_ddot, xdes_dddot, xdes_ddddot = x_des[0][:5, time_idx]  # X-axis
        ydes, ydes_dot, ydes_ddot, ydes_dddot, ydes_ddddot = x_des[1][:5, time_idx]  # Y-axis
        zdes, zdes_dot, zdes_ddot, zdes_dddot, zdes_ddddot = x_des[2][:5, time_idx]  # Z-axis
        psi_des, psi_des_dot = x_des[3][:2, time_idx]
        
        # Errors
        ex = np.array([rx, ry, rz]) - np.array([xdes, ydes, zdes])
        ev = np.array([vx, vy, vz]) - np.array([xdes_dot, ydes_dot, zdes_dot])
        
        # Total thrust
        Fd = (-(self.kx * ex) - (self.kv * ev)) + (m * np.array([xdes_ddot, ydes_ddot, zdes_ddot])) + (m * g * np.array([0, 0, 1]))
        zB = rotation[:, 2]
        Fz = np.dot(Fd, zB)
        
        acc_des = np.array([xdes_ddot, ydes_ddot, zdes_ddot + g]).T
        acc = (Fz*zB)/m - g * np.array([0, 0, 1])
        ea = acc - acc_des
        
        jerk = np.array([xdes_dddot, ydes_dddot, zdes_dddot])
        dFd = (-(self.kx * ex) - (self.kv * ea) + jerk)
        dFz = dFd @ zB
    
        # Desired attitudes
        if np.linalg.norm(Fd) < 1e-8:
            raise ValueError("Fd is too small")
        zB_des = Fd / np.linalg.norm(Fd)
        xC_des = np.array([np.cos(psi_des), np.sin(psi_des), 0]).T
        zBxC = np.cross(zB_des, xC_des)
        yB_des = zBxC / np.linalg.norm(zBxC)
        xB_des = np.cross(yB_des, zB_des)
        Rd = np.c_[xB_des.T,yB_des.T,zB_des.T]

        u1 = m * np.linalg.norm(acc_des)
        h_omega = (m / u1) * (jerk - (zB_des @ jerk) * zB_des)
        p = -h_omega @ yB_des
        q = h_omega @ xB_des
        r = psi_des_dot * np.array([0, 0, 1]) @ zB_des
        omega_des = np.hstack([p,q,r])

        # Errors
        eR = 0.5 * self.vee_map(Rd.T @ rotation - rotation.T @ Rd)
        e_omega = omega - omega_des

        # Moments
        M = (-self.kR * eR - self.komega * e_omega)

        return Fz, M

    def actuator_allocation(self, Fz, M, N_inv):
        """Compute motor inputs from force and moment."""
        return N_inv @ np.array([Fz, *M])