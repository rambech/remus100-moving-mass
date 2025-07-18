import numpy as np

import gnc


class Woolsey():
    """
    Remus vehicle class with moving mass, Hamiltonian formulation by Woolsey and Leonard (2002), https://doi.org/10.1109/ACC.2002.1025217
    """

    def __init__(self, dt: float = 0.02):
        self.dt = dt

        # Constants
        gravity = 9.81   # acceleration of gravity [m/s^2]
        self.rho = 1026  # density of water [kg/m^3]

        # Vehicle dimensions
        self.L = 1.6    # Length   [m]
        self.d = 0.19   # Diameter [m]
        self.dof = 2    # Number of DOFs

        # Hydrodynamics (Fossen 2021, Section 8.4.2)
        self.S = 0.7 * self.L * self.d    # S = 70% of rectangle L * diam
        a = self.L/2                      # semi-axes
        b = self.d/2

        # Moving mass bounds
        x_pmax = 0.05    # Max fwd. position of moving mass [m]
        x_pmin = -0.05   # Max bckwd. position of mm. [m]
        self.upper_p = np.array([x_pmax, 0, 0.05])
        self.lower_p = np.array([x_pmin, 0, 0.05])

        # Parasitic drag coefficient CD_0, i.e. zero lift and alpha = 0
        # F_drag = 0.5 * rho * Cd * (pi * b^2)
        # F_drag = 0.5 * rho * CD_0 * S
        Cd = 0.42                              # from Allen et al. (2000)
        self.CD_0 = Cd * np.pi * b**2 / self.S

        # Vehicle parameters
        self.m = 4/3 * np.pi * self.rho * a * b**2   # mass of spheriod
        self.m_s = 5/6 * self.m         # Rigid body mass
        self.m_p = self.m - self.m_s    # Moving mass is one-sixth of the total mass
        # Location of static mass w.r.t CO
        self.r_s = np.array([0.0, 0.0, 0.0])
        self.r_b = np.array([0.0, 0.0, 0.0])   # CB w.r.t. CO

        # Approximate inertia dyadic for a spheroid (Fossen 2021, Chapter 8)
        I_x = (2/5) * self.m_s * b**2
        I_y = (1/5) * self.m_s * (a**2 + b**2)
        I_z = I_y
        self.I_b = np.diag([I_x, I_y, I_z])

        # Weight and buoyancy
        self.W = self.m * gravity
        self.B = self.W

        # Added moment of inertia in roll: A44 = r44 * Ix
        R44 = 0.3
        M_A_44 = R44 * I_x

        # Lamb's k-factors
        e = np.sqrt(1-(b/a)**2)
        alpha_0 = (2 * (1-e**2)/pow(e, 3)) * (0.5 * np.log((1+e)/(1-e)) - e)
        beta_0 = 1/(e**2) - (1-e**2) / (2*pow(e, 3)) * np.log((1+e)/(1-e))

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0 / (2 - beta_0)
        k_prime = pow(e, 4) * (beta_0-alpha_0) / (
            (2-e**2) * (2*e**2 - (2-e**2) * (beta_0-alpha_0)))

        # Added mass system matrix expressed in the CO
        self.M_A = np.diag(
            [self.m*k1, self.m*k2, self.m*k2,
             M_A_44, k_prime*I_y, k_prime*I_y,
             0, 0, 0]
        )

        # System mass matrix
        # self.M = M_S + self.M_A

        # Low-speed linear damping matrix parameters
        self.T_surge = 20           # time constant in surge [s]
        self.T_sway = 20            # time constant in sway [s]
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3        # relative damping ratio in roll
        self.zeta_pitch = 0.8       # relative damping ratio in pitch
        self.T_yaw = 1              # time constant in yaw [s]

    def step(self, eta, r_p, nu, v_p, u_control):
        """
        Moving mass Remus 100 step method, Woolsey and Leonard (2002) Hamiltonian formulation

        Parameters
        ----------
            eta: np.ndarray
                vehicle pose vector, eta = [x, y, z, phi, theta, psi]
            r_p: np.ndarray
                moving mass position vector, r_p = [x_p, y_p, z_p]
            nu: np.ndarray
                vehicle velocity vector, nu = [u, v, w, p, q, r]
            v_p: np.ndarray
                moving mass velocity vector, v_p = [u_p, v_p, w_p]
            u_control: np.ndarray
                control forces vector, u_control = [tau_X, tau_Xp]

        Returns
        -------
            eta: np.ndarray
                vehicle pose vector, eta = [x, y, z, phi, theta, psi]
            r_p: np.ndarray
                moving mass position vector, r_p = [x_p, y_p, z_p]
            r_p_dot: np.ndarray
                change in moving mass position w.r.t CO, r_p_dot = [x_p_dot, y_p_dot, z_p_dot]
            nu: np.ndarray
                vehicle velocity vector, nu = [u, v, w, p, q, r]
            v_p: np.array
                moving mass velocity vector, v_p = [u_p, v_p, w_p]
        """

        alpha = np.arctan2(nu[2], nu[0])
        U = np.linalg.norm(nu[0:3])
        g = 9.81

        r_g = (self.m_s*self.r_s + self.m_p*r_p)/self.m

        # Added mass system matrix expressed in the CO
        # For an ellipsoid A_12 and A_21 are zero
        A_11 = self.M_A[0:3, 0:3]
        A_12 = self.M_A[0:3, 3:6]
        A_21 = self.M_A[3:6, 0:3]
        A_22 = self.M_A[3:6, 3:6]

        # Moving mass weight matrix
        # M = [       mI3 + A11                 -m_sS(r_g) - m_pS(r_p) + A12       mpI3
        #       m_sS(r_g) + m_pS(r_p) + A21        Ib - m_pS(r_p)**2 + A22      m_pS(r_p)
        #               mpI3                              -m_pS(r_p)               mpI3   ]
        M = np.block([
            [
                self.m*np.eye(3) + A_11,
                -self.m_s*gnc.linalg.Smtrx(r_g) -
                self.m_p*gnc.linalg.Smtrx(r_p) + A_12,
                self.m_p*np.eye(3)
            ],
            [
                self.m_s*gnc.linalg.Smtrx(r_g) +
                self.m_p*gnc.linalg.Smtrx(r_p) + A_21,
                self.I_b - self.m_p *
                gnc.linalg.Smtrx(r_p).dot(gnc.linalg.Smtrx(r_p))
                + A_22,
                self.m_p*gnc.linalg.Smtrx(r_p)
            ],
            [
                self.m_p*np.eye(3),
                -self.m_p*gnc.linalg.Smtrx(r_p),
                self.m_p*np.eye(3)
            ]
        ])
        Minv = np.linalg.inv(M)

        # Hydrodynamic damping
        # Natural frequencies in roll and pitch
        w_roll = np.sqrt(self.W * (r_g[2]-self.r_b[2]) /
                         M[3, 3])
        w_pitch = np.sqrt(self.W * (r_g[2]-self.r_b[2]) /
                          M[4, 4])

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -M[0, 0] / self.T_surge
        Yv = -M[1, 1] / self.T_sway
        Zw = -M[2, 2] / self.T_heave
        Kp = -M[3, 3] * 2 * self.zeta_roll * w_roll
        Mq = -M[4, 4] * 2 * self.zeta_pitch * w_pitch
        Nr = -M[5, 5] / self.T_yaw
        Xu_p = 0
        Yv_p = 0
        Zw_p = 0

        D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr, Xu_p, Yv_p, Zw_p])

        # Vanish at high speed where quadratic drag and lift forces dominates
        D[0, 0] = D[0, 0] * np.exp(-3*U)
        D[1, 1] = D[1, 1] * np.exp(-3*U)

        tau_damp = -np.matmul(D, np.concatenate([nu, v_p]))

        # Nonlinear yaw damping
        tau_damp[5] = tau_damp[5] - 10 * \
            D[5, 5] * abs(nu[5]) * nu[5]

        # State derivatives (with dimension)
        tau_liftdrag = gnc.forceLiftDrag(
            self.d, self.S, self.CD_0, alpha, U
        )
        tau_liftdrag = np.concatenate([tau_liftdrag, [0, 0, 0]])

        # Control forces
        tau_prime = np.zeros(9)
        tau_prime[0] = u_control[0]
        tau_prime[6] = u_control[1]

        # Do not add extra force when at the ends
        # NOTE: Does introduce some chattering
        if r_p[0] >= self.upper_p[0] and u_control[1] > 0:
            tau_prime[6] = 0
        elif r_p[0] <= self.lower_p[0] and u_control[1] < 0:
            tau_prime[6] = 0

        # Sum of control and environmental forces
        sum_tau = (
            tau_prime
            + tau_damp
            + tau_liftdrag
        )

        R = gnc.linalg.Rzyx(eta[3], eta[4], eta[5])
        Sr_g = gnc.linalg.Smtrx(r_g)
        # Comment in line below to be equvalent to Rambech et al. 2025
        # Sr_g = np.zeros((3, 3))
        Sr_p = gnc.linalg.Smtrx(r_p)
        I3 = np.eye(3)
        k = np.array([0, 0, 1])  # z unit vector

        # Angular vehicle momentum
        PI = (
            (self.I_b + A_22).dot(nu[3:6])
            + (self.m_s*Sr_g + A_21).dot(nu[0:3])
            + Sr_p.dot(self.m_p*v_p)
        )
        # Linear vehicle momentum
        P = (
            (-self.m_s*Sr_g + A_12).dot(nu[3:6])
            + (self.m_s*I3 + A_11).dot(nu[0:3])
            + self.m_p*v_p
        )
        # Linear moving mass momentum
        P_p = self.m_p*v_p

        # Vehicle torque
        PI_dot = (
            gnc.linalg.Smtrx(PI).dot(nu[3:6])
            + gnc.linalg.Smtrx(P).dot(nu[0:3])
            + self.m_s*g*Sr_g.dot(R.T.dot(k))
            + self.m_p*g*Sr_p.dot(R.T.dot(k))
            + sum_tau[3:6]
        )
        # Vehicle forces
        P_dot = gnc.linalg.Smtrx(P).dot(nu[3:6]) + sum_tau[0:3]
        # Moving mass forces
        P_p_dot = (
            gnc.linalg.Smtrx(P_p).dot(nu[3:6])
            + self.m_p*g*(R.T.dot(k))
            + sum_tau[6:9]
            - self.m_p*g*(R.T.dot(k))  # "Control force" counteracting gravity
        )

        # Kinetic step
        nu_prime_dot = Minv.dot(np.concatenate([P_dot, PI_dot, P_p_dot]))
        v_dot = nu_prime_dot[0:3]
        Omega_dot = nu_prime_dot[3:6]
        v_p_dot = nu_prime_dot[6:9]
        nu[0:3] = nu[0:3] + v_dot * self.dt
        nu[3:6] = nu[3:6] + Omega_dot * self.dt
        v_p = v_p + v_p_dot * self.dt

        temp_r_p_dot = v_p - nu[0:3] - gnc.linalg.Smtrx(nu[3:6]).dot(r_p)
        # If at one of the ends, simply follow the vessel
        v_p_equals_v = nu[0:3] + gnc.linalg.Smtrx(nu[3:6]).dot(r_p)
        if ((r_p[0] >= self.upper_p[0] and temp_r_p_dot[0] > 0) or
                (r_p[0] <= self.lower_p[0] and temp_r_p_dot[0] < 0)):
            v_p = nu[0:3] + gnc.linalg.Smtrx(nu[3:6]).dot(r_p)
        else:
            v_p[1:] = v_p_equals_v[1:]

        r_p_dot = v_p - nu[0:3] - gnc.linalg.Smtrx(nu[3:6]).dot(r_p)
        # Round out float point inaccuracies of the above lines
        r_p_dot[1:] = np.round(r_p_dot[1:], 8)

        # Kinematic step
        p_dot = R.dot(nu[0:3])
        R_dot = gnc.linalg.Tzyx(eta[3], eta[4]).dot(nu[3:6])
        eta[0:3] = eta[0:3] + p_dot * self.dt
        eta[3:6] = eta[3:6] + R_dot * self.dt
        r_p = r_p + r_p_dot * self.dt

        # Saturate moving mass position at the ends
        if r_p[0] >= self.upper_p[0] and u_control[1] > 0:
            r_p[0] = self.upper_p[0]
        elif r_p[0] <= self.lower_p[0] and u_control[1] < 0:
            r_p[0] = self.lower_p[0]

        return eta, r_p, r_p_dot, nu, v_p
