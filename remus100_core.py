#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remus100_core.py: Core functionalities for the REMUS 100 AUV model,
including mathematical utilities, dynamics, controller, and simulator.
"""
import numpy as np
import math


CURRENT_MODEL_SIM_EXACT = "inertial_exact"

# ==============================================================================
# SECTION 1: Math and Attitude Utilities
# ==============================================================================

def euler_to_quaternion(phi, theta, psi):
    """Converts Euler angles to a quaternion."""
    cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
    cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
    cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def quaternion_to_euler(q):
    """Converts a quaternion to Euler angles."""
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-8:
        return np.zeros(3)
    q = q / q_norm
    w, x, y, z = q
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        theta = np.copysign(np.pi / 2, sinp)
        phi = 2 * np.arctan2(x, w)
        psi = 0.0
    else:
        theta = np.arcsin(sinp)
        phi = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        psi = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    return np.array([phi, theta, psi])

def quaternion_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def rotation_matrix_from_euler(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    R = np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth, cth * sphi, cth * cphi]])
    if R.ndim == 3:
        return R.transpose(2, 0, 1)
    return R

def rotation_matrix_from_quaternion(q):
    """Generates a rotation matrix from a quaternion."""
    norm = np.linalg.norm(q)
    q = q / norm if norm > 1e-8 else np.array([1., 0., 0., 0.])
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]])

def angular_velocity_transformation(phi, theta):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    if abs(cth) < 1e-8: cth = np.copysign(1e-8, cth)
    return np.array([[1, sphi * sth / cth, cphi * sth / cth], [0, cphi, -sphi], [0, sphi / cth, cphi / cth]])

def smallest_signed_angle(angle):
    """Computes the smallest signed angle in the range [-pi, pi]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

def _rk4_step(f, x, u, t, dt, **kwargs):
    """Performs a single step of the 4th-order Runge-Kutta method."""
    k1 = f(x, u, t, **kwargs)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt, **kwargs)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt, **kwargs)
    k4 = f(x + dt * k3, u, t + dt, **kwargs)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def integrate(f, x, u, t, dt, method='rk4', **kwargs):
    """Integrates the state vector over a single time step."""
    if method == 'rk4':
        return _rk4_step(f, x, u, t, dt, **kwargs)
    else:
        return x + dt * f(x, u, t, **kwargs)

# ==============================================================================
# SECTION 2: AUV Dynamics Model
# ==============================================================================

class Remus100Dynamics:
    """6-DOF dynamics model for the REMUS 100 AUV."""
    def __init__(self):
        self.rho, g = 1026, 9.81
        self.L, self.diam = 1.6, 0.19
        self.r_bg = np.array([0, 0, 0.02])
        self.r_bb = np.array([0, 0, 0])

        a, b = self.L / 2, self.diam / 2
        m = (4 / 3) * math.pi * self.rho * a * b**2
        self.W, self.B = m * g, m * g
        Ix = (2 / 5) * m * b**2
        Iy = (1 / 5) * m * (a**2 + b**2)
        Iz = (1 / 5) * m * (a**2 + b**2)

        MRB_CG = np.diag([m, m, m, Ix, Iy, Iz])
        H = np.identity(6)
        H[0:3, 3:6] = -self._skew(self.r_bg)
        self.MRB = H.T @ MRB_CG @ H

        e = math.sqrt(1 - (b / a)**2)
        alpha_0 = (2 * (1 - e**2) / e**3) * (0.5 * math.log((1 + e) / (1 - e)) - e)
        beta_0 = 1 / e**2 - (1 - e**2) / (2 * e**3) * math.log((1 + e) / (1 - e))
        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0 / (2 - beta_0)
        k_prime = e**4 * (beta_0 - alpha_0) / ((2 - e**2) * (2 * e**2 - (2 - e**2) * (beta_0 - alpha_0)))
        MA_44 = 0.3 * Ix
        self.MA = np.diag([m * k1, m * k2, m * k2, MA_44, k_prime * Iy, k_prime * Iy])

        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)
        self.w_roll = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) / self.M[3, 3])
        self.w_pitch = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) / self.M[4, 4])
        self.T_surge, self.T_sway, self.T_heave, self.T_yaw = 20, 20, 20, 1
        self.zeta_roll, self.zeta_pitch = 0.3, 0.8

        S_fin = 0.00665
        self.A_r, self.A_s = 2 * S_fin, 2 * S_fin
        self.CL_delta_r, self.CL_delta_s = 0.5, 0.7
        self.x_r, self.x_s = -a, -a

        self.D_prop, self.t_prop = 0.14, 0.1
        self.KT_0, self.KQ_0 = 0.4566, 0.0700
        self.KT_max, self.KQ_max = 0.1798, 0.0312
        self.Ja_max = 0.6632

        self.T_delta, self.T_n = 0.1, 1.0

        self.S = 0.7 * self.L * self.diam
        Cd = 0.42
        self.CD_0 = Cd * math.pi * b**2 / self.S

    def _current_inertial_vector(self, V_c=0, beta_c=0, w_c=0, v_c_inertial=None):
        """Return inertial-frame current velocity."""
        if v_c_inertial is not None:
            return np.asarray(v_c_inertial, dtype=float)
        return np.array([
            V_c * math.cos(beta_c),
            V_c * math.sin(beta_c),
            w_c,
        ], dtype=float)

    def body_current_velocity(self, eta, R=None, V_c=0, beta_c=0, w_c=0, v_c_inertial=None):
        """Return exact body-frame current velocity R^T v_c^n."""
        v_c_n = self._current_inertial_vector(V_c, beta_c, w_c, v_c_inertial)
        if R is None:
            R = rotation_matrix_from_euler(eta[3], eta[4], eta[5])
        return np.asarray(R, dtype=float).T @ v_c_n

    def current_twist(self, eta, R=None, V_c=0, beta_c=0, w_c=0, v_c_inertial=None):
        """Return 6D current twist [v_c_body, 0, 0, 0]."""
        v_c_body = self.body_current_velocity(
            eta, R=R, V_c=V_c, beta_c=beta_c, w_c=w_c, v_c_inertial=v_c_inertial,
        )
        return np.array([v_c_body[0], v_c_body[1], v_c_body[2], 0.0, 0.0, 0.0])

    def relative_velocity(self, eta, nu_total, R=None, V_c=0, beta_c=0, w_c=0, v_c_inertial=None):
        """Convert total body-frame velocity to relative water velocity."""
        return np.asarray(nu_total, dtype=float) - self.current_twist(
            eta, R=R, V_c=V_c, beta_c=beta_c, w_c=w_c, v_c_inertial=v_c_inertial,
        )

    def total_velocity(self, eta, nu_r, R=None, V_c=0, beta_c=0, w_c=0, v_c_inertial=None):
        """Convert relative water velocity back to total body-frame velocity."""
        return np.asarray(nu_r, dtype=float) + self.current_twist(
            eta, R=R, V_c=V_c, beta_c=beta_c, w_c=w_c, v_c_inertial=v_c_inertial,
        )

    def _control_forces(self, nu_r, u_actual, prop_inflow_speed):
        """Compute control forces from fins and propeller."""
        delta_r, delta_s, n = u_actual
        n_rps = n / 60.0
        Va = 0.944 * abs(prop_inflow_speed)
        if n_rps > 0:
            X_prop = self.rho * self.D_prop**4 * (
                self.KT_0 * abs(n_rps) * n_rps
                + (self.KT_max - self.KT_0) / self.Ja_max
                * (Va / self.D_prop) * abs(n_rps)
            )
            K_prop = self.rho * self.D_prop**5 * (
                self.KQ_0 * abs(n_rps) * n_rps
                + (self.KQ_max - self.KQ_0) / self.Ja_max
                * (Va / self.D_prop) * abs(n_rps)
            )
        else:
            X_prop = self.rho * self.D_prop**4 * self.KT_0 * abs(n_rps) * n_rps
            K_prop = self.rho * self.D_prop**5 * self.KQ_0 * abs(n_rps) * n_rps
        K_prop /= 10.0

        U_rh = math.sqrt(nu_r[0]**2 + nu_r[1]**2)
        U_rv = math.sqrt(nu_r[0]**2 + nu_r[2]**2)
        X_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_r**2
        X_s = -0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_s**2
        Y_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_r
        Z_s = -0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_s
        return np.array([
            (1 - self.t_prop) * X_prop + X_r + X_s,
            Y_r,
            Z_s,
            K_prop,
            -self.x_s * Z_s,
            self.x_r * Y_r,
        ])

    def _relative_rhs(self, eta, nu_r, u_actual, u_control, prop_inflow_speed):
        """Compute relative-velocity dynamics shared by both current models."""
        delta_r, delta_s, n = u_actual

        # Coriolis and centripetal matrices
        CRB = self._m2c(self.MRB, nu_r)
        CA = self._m2c(self.MA, nu_r)
        CA[4, 0] = 0; CA[0, 4] = 0
        CA[4, 2] = 0; CA[2, 4] = 0
        CA[5, 0] = 0; CA[0, 5] = 0
        CA[5, 1] = 0; CA[1, 5] = 0
        C = CRB + CA

        # Damping matrix
        U_r = np.linalg.norm(nu_r[0:3])
        D = np.diag([
            self.M[0, 0] / self.T_surge, self.M[1, 1] / self.T_sway,
            self.M[2, 2] / self.T_heave, self.M[3, 3] * 2 * self.zeta_roll * self.w_roll,
            self.M[4, 4] * 2 * self.zeta_pitch * self.w_pitch, self.M[5, 5] / self.T_yaw])
        D[0, 0] = D[0, 0] * math.exp(-3 * U_r)
        D[1, 1] = D[1, 1] * math.exp(-3 * U_r)

        # Gravitational and buoyancy forces
        g_vec = self._g_forces(eta[3], eta[4])

        # Control forces
        tau_control = self._control_forces(nu_r, u_actual, prop_inflow_speed)

        # Hydrodynamic forces
        alpha = math.atan2(nu_r[2], nu_r[0])
        tau_liftdrag = self._force_lift_drag(alpha, U_r)
        tau_crossflow = self._cross_flow_drag(nu_r)

        # Sum of forces and state derivatives
        tau_sum = tau_control + tau_liftdrag + tau_crossflow - (C + D) @ nu_r - g_vec
        nu_r_dot = self.Minv @ tau_sum

        delta_r_c, delta_s_c, n_c = u_control
        u_dot = np.array([(delta_r_c - delta_r) / self.T_delta,
                          (delta_s_c - delta_s) / self.T_delta,
                          (n_c - n) / self.T_n])
        return nu_r_dot, u_dot

    def compute_derivatives(self, eta, nu, u_actual, u_control,
                             V_c=0, beta_c=0, w_c=0, v_c_inertial=None):
        """Compute velocity and actuator derivatives.

        The velocity state ``nu`` is the total body-frame velocity.
        """
        v_c_n = self._current_inertial_vector(V_c, beta_c, w_c, v_c_inertial)
        R = rotation_matrix_from_euler(eta[3], eta[4], eta[5])
        nu = np.asarray(nu, dtype=float)
        nu_c = self.current_twist(
            eta,
            R=R,
            V_c=V_c,
            beta_c=beta_c,
            w_c=w_c,
            v_c_inertial=v_c_n,
        )
        d_v_c_body = -np.cross(nu[3:6], nu_c[:3])
        Dnu_c = np.concatenate([d_v_c_body, np.zeros(3, dtype=float)])
        nu_r = nu - nu_c
        nu_r_dot, u_dot = self._relative_rhs(
            # Propeller inflow should follow the axial relative flow, not the
            # inertial-frame total speed magnitude.
            eta, nu_r, u_actual, u_control, prop_inflow_speed=nu_r[0],
        )
        nu_dot = Dnu_c + nu_r_dot
        return nu_dot, u_dot

    def _skew(self, v):
        """Creates a skew-symmetric matrix from a 3-element vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _m2c(self, M, nu):
        """Computes the Coriolis matrix from a mass matrix and velocity vector."""
        M = 0.5 * (M + M.T)
        S1 = self._skew(M[0:3, 0:3] @ nu[0:3] + M[0:3, 3:6] @ nu[3:6])
        S2 = self._skew(M[3:6, 0:3] @ nu[0:3] + M[3:6, 3:6] @ nu[3:6])
        C = np.zeros((6, 6))
        C[0:3, 3:6], C[3:6, 0:3], C[3:6, 3:6] = -S1, -S1, -S2
        return C

    def _g_forces(self, phi, theta):
        """Computes gravitational and buoyancy forces and moments."""
        sth, cth = math.sin(theta), math.cos(theta)
        sphi, cphi = math.sin(phi), math.cos(phi)
        W, B, rbg, rbb = self.W, self.B, self.r_bg, self.r_bb
        return np.array([(W - B) * sth,
                         -(W - B) * cth * sphi,
                         -(W - B) * cth * cphi,
                         -(rbg[1] * W - rbb[1] * B) * cth * cphi + (rbg[2] * W - rbb[2] * B) * cth * sphi,
                         (rbg[2] * W - rbb[2] * B) * sth + (rbg[0] * W - rbb[0] * B) * cth * cphi,
                         -(rbg[0] * W - rbb[0] * B) * cth * sphi - (rbg[1] * W - rbb[1] * B) * sth])

    def _force_lift_drag(self, alpha, U_r):
        """Computes lift and drag forces on the vehicle body."""
        AR = self.diam**2 / self.S
        e = 0.7
        CL_alpha = math.pi * AR / (1 + math.sqrt(1 + (AR / 2)**2))
        CL = CL_alpha * alpha
        CD = self.CD_0 + CL**2 / (math.pi * e * AR)
        F_drag = 0.5 * self.rho * U_r**2 * self.S * CD
        F_lift = 0.5 * self.rho * U_r**2 * self.S * CL
        tau = np.zeros(6)
        tau[0] = math.cos(alpha) * (-F_drag) - math.sin(alpha) * (-F_lift)
        tau[2] = math.sin(alpha) * (-F_drag) + math.cos(alpha) * (-F_lift)
        return tau

    def _cross_flow_drag(self, nu_r):
        """Computes cross-flow drag forces."""
        x = np.array([0.0109,0.1766,0.3530,0.4519,0.4728,0.4929,0.4933,0.5585,0.6464,0.8336,
                      0.9880,1.3081,1.6392,1.8600,2.3129,2.6000,3.0088,3.4508, 3.7379,4.0031])
        y = np.array([1.9661,1.9657,1.8976,1.7872,1.5837,1.2786,1.2108,1.0836,0.9986,0.8796,
                      0.8284,0.7599,0.6914,0.6571,0.6307,0.5962,0.5868,0.5859,0.5599,0.5593])
        Cd_2D = np.interp(self.diam / (2 * self.diam), x, y) # # B/(2T) for cylinder
        Yh, Nh = 0, 0
        n = 20
        dx = self.L / n
        for i in range(n + 1):
            x_pos = -self.L / 2 + i * dx
            Ucf = abs(nu_r[1] + x_pos * nu_r[5]) * (nu_r[1] + x_pos * nu_r[5])
            Yh -= 0.5 * self.rho * self.diam * Cd_2D * Ucf * dx
            Nh -= 0.5 * self.rho * self.diam * Cd_2D * x_pos * Ucf * dx
        tau = np.zeros(6)
        tau[1], tau[5] = Yh, Nh
        return tau

# ==============================================================================
# SECTION 3: AUV Controller
# ==============================================================================

class Remus100Controller:
    """ Depth and Heading autopilot for REMUS 100 AUV. """
    def __init__(self):
        self.D2R = math.pi / 180
        
        # Depth autopilot parameters
        self.wn_d_z, self.Kp_z, self.T_z = 0.02, 0.1, 100.0
        self.Kp_theta, self.Kd_theta, self.Ki_theta, self.K_w = 5.0, 2.0, 0.3, 5.0
        
        # Heading autopilot (SMC) parameters
        self.wn_d, self.zeta_d, self.r_max = 0.1, 1.0, 5.0 * self.D2R
        self.lam, self.phi_b, self.K_d, self.K_sigma = 0.1, 0.1, 0.5, 0.05
        self.K_nomoto, self.T_nomoto = 5.0 / 20.0, 1.0
        
        self.reset()

    def set_control_gains(self, gains):
        for k, v in gains.items():
            if hasattr(self, k): setattr(self, k, v)

    def reset(self):
        self.z_d, self.z_int, self.theta_int = 0, 0, 0
        self.psi_d, self.r_d, self.a_d, self.e_psi_int = 0, 0, 0, 0
    
    def step(self, eta, nu, dt, z_ref, psi_ref, n_ref):
        z, theta, psi = eta[2], eta[4], eta[5]
        w, q, r = nu[2], nu[4], nu[5]
        
        # ----------------------------------------------------------------------
        # Depth Autopilot (Successive Loop Closure)
        # ----------------------------------------------------------------------
        # LP filtered desired depth command
        self.z_d = math.exp(-dt * self.wn_d_z) * self.z_d + (1 - math.exp(-dt * self.wn_d_z)) * z_ref
        
        # PI controller for outer loop (depth) -> desired pitch
        depth_error = z - self.z_d
        self.z_int = np.clip(self.z_int + dt * depth_error, -50.0, 50.0)
        theta_d = self.Kp_z * (depth_error + (1 / self.T_z) * self.z_int)
        
        # PID controller for inner loop (pitch) -> stern plane command
        pitch_error = smallest_signed_angle(theta - theta_d)
        max_theta_int = 15.0 * self.D2R
        if abs(pitch_error) < abs(smallest_signed_angle(pitch_error - self.theta_int * dt)):
            self.theta_int += dt * pitch_error
            self.theta_int = np.clip(self.theta_int, -max_theta_int, max_theta_int)
        delta_s = -(self.Kp_theta * pitch_error + self.Kd_theta * q + self.Ki_theta * self.theta_int + self.K_w * w)
        
        # ----------------------------------------------------------------------
        # Heading Autopilot (Integral SMC)
        # ----------------------------------------------------------------------
        psi_ref_rad = psi_ref * self.D2R        
        
        # 3rd-order reference model
        a_d_dot = self.wn_d**3 * (psi_ref_rad - self.psi_d) - (2 * self.zeta_d + 1) * self.wn_d**2 * self.r_d - (2 * self.zeta_d + 1) * self.wn_d * self.a_d
        self.a_d += dt * a_d_dot
        self.r_d = np.clip(self.r_d + dt * self.a_d, -self.r_max, self.r_max)
        self.psi_d += dt * self.r_d
        
        # Sliding surface and control law
        heading_error = smallest_signed_angle(psi - self.psi_d)
        rate_error = r - self.r_d
        sigma = rate_error + 2 * self.lam * heading_error + self.lam**2 * self.e_psi_int
        
        v_r_dot = self.a_d - 2 * self.lam * rate_error - self.lam**2 * heading_error
        v_r = self.r_d - 2 * self.lam * heading_error - self.lam**2 * self.e_psi_int

        sat_sigma = np.sign(sigma) if abs(sigma / self.phi_b) > 1.0 else sigma / self.phi_b
        delta_r = (self.T_nomoto * v_r_dot + v_r - self.K_d * sigma - self.K_sigma * sat_sigma) / self.K_nomoto

        max_heading_int = 10.0 * self.D2R
        if abs(self.e_psi_int) < max_heading_int or np.sign(heading_error) != np.sign(self.e_psi_int):
            self.e_psi_int += dt * heading_error
            self.e_psi_int = np.clip(self.e_psi_int, -max_heading_int, max_heading_int)
            
        return np.array([delta_r, -delta_s, n_ref])

# ==============================================================================
# SECTION 4: AUV Simulator
# ==============================================================================

class Remus100Simulator:
    """
    High-fidelity AUV simulator using a unified state vector and quaternion-based
    attitude representation for numerical stability and accuracy.
    """
    def __init__(self, dynamics, dt=0.02, integrator='rk4'):
        self.dynamics = dynamics
        self.dt = dt
        self.integrator_method = integrator
        self.reset()

    def reset(self, eta0=None, nu0=None, v_c_inertial=None):
        """Resets the simulator to an initial state."""
        self.eta = np.zeros(6) if eta0 is None else np.array(eta0, dtype=float)
        self.nu = np.zeros(6) if nu0 is None else np.array(nu0, dtype=float)
        self.nu_r = self.nu.copy()
        self.u_actual = np.zeros(3)
        self.time = 0.0
        self.quaternion = euler_to_quaternion(self.eta[3], self.eta[4], self.eta[5])
        self.current_v_c_inertial = np.zeros(3) if v_c_inertial is None else np.array(v_c_inertial, dtype=float)
        R = rotation_matrix_from_quaternion(self.quaternion)
        self.nu_r = self.dynamics.relative_velocity(
            self.eta, self.nu, R=R, v_c_inertial=self.current_v_c_inertial,
        )

    def _state_derivative(self, x, u_control, t, V_c=0, beta_c=0, w_c=0, v_c_inertial=None):
        """
        Computes the derivative of the complete 16-DOF state vector
        x = [position, quaternion, velocity_state, actuator_states].
        """
        pos, quat, nu_r_state, u_act = x[0:3], x[3:7], x[7:13], x[13:16]

        norm = np.linalg.norm(quat)
        quat = quat / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0])

        intermediate_euler_angles = quaternion_to_euler(quat)
        mock_eta = np.concatenate([pos, intermediate_euler_angles])
        
        nu_r_dot, u_dot = self.dynamics._relative_rhs(
            mock_eta, nu_r_state, u_act, u_control, prop_inflow_speed=nu_r_state[0]
        )

        v_c_n = self.dynamics._current_inertial_vector(
            V_c=V_c, beta_c=beta_c, w_c=w_c, v_c_inertial=v_c_inertial,
        )
        R = rotation_matrix_from_quaternion(quat)
        pos_dot = R @ nu_r_state[0:3] + v_c_n

        p, q, r = nu_r_state[3:6]
        omega_quat = np.array([0, p, q, r])
        quat_dot = 0.5 * quaternion_multiply(quat, omega_quat)

        return np.concatenate([pos_dot, quat_dot, nu_r_dot, u_dot])

    def step(self, u_control, V_c=0, beta_c=0, w_c=0, v_c_inertial=None):
        """Advances the simulation by one time step."""
        pos = self.eta[0:3]
        v_c_n = self.dynamics._current_inertial_vector(
            V_c=V_c, beta_c=beta_c, w_c=w_c, v_c_inertial=v_c_inertial,
        )
        R = rotation_matrix_from_quaternion(self.quaternion)
        # Preserve total body velocity across step-wise current updates by
        # re-expressing the carried state in the new relative-current frame.
        nu_r_start = self.dynamics.relative_velocity(
            self.eta, self.nu, R=R, v_c_inertial=v_c_n,
        )
        current_state = np.concatenate([pos, self.quaternion, nu_r_start, self.u_actual])

        x_new = integrate(self._state_derivative, current_state, u_control, self.time, self.dt,
                          method=self.integrator_method, V_c=V_c, beta_c=beta_c, w_c=w_c,
                          v_c_inertial=v_c_n)

        # Update state vectors from the new integrated state
        self.eta[0:3] = x_new[0:3]
        self.quaternion = x_new[3:7]
        self.nu_r = x_new[7:13]
        self.u_actual = x_new[13:16]

        # Normalize quaternion to prevent numerical drift
        self.quaternion /= np.linalg.norm(self.quaternion)

        # Update Euler angles from the new quaternion
        self.eta[3:6] = quaternion_to_euler(self.quaternion)
        R = rotation_matrix_from_quaternion(self.quaternion)
        self.current_v_c_inertial = v_c_n

        self.nu = self.dynamics.total_velocity(
            self.eta, self.nu_r, R=R, v_c_inertial=v_c_n,
        )

        self.time += self.dt

        # Clip actuator states to physical limits
        delta_max = 15 * math.pi / 180
        delta_max_n = 1525 
        self.u_actual[0] = np.clip(self.u_actual[0], -delta_max, delta_max) # Rudder
        self.u_actual[1] = np.clip(self.u_actual[1], -delta_max, delta_max) # Stern plane
        self.u_actual[2] = np.clip(self.u_actual[2], -delta_max_n, delta_max_n) # Propeller RPM

        return self.eta, self.nu
