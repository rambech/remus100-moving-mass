"""
Remus vehicle class

Author: @rambech
Based on the work of Thor Inge Fossen (aka @cybergalactic)
"""

# General imports
import numpy as np

# Specific imports
import gnc


class Rambech():
    """
    Remus vehicle class with moving mass, Newton-Euler formulation by Rambech et al. (2025)
    """

    def __init__(self, dt: float = 0.02):
        super().__init__()
        self.dt = dt

        self._init_model()

    def _init_model(self):
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
        self.m_s = 5/6 * self.m
        self.m_p = self.m - self.m_s    # Moving mass is one-sixth of the total mass
        # Location of static mass w.r.t CO
        self.r_s = np.array([0.0, 0.0, 0.0])
        self.r_b = np.array([0.0, 0.0, 0.0])   # CB w.r.t. CO

        # Approximate inertia dyadic for a spheroid (Fossen 2021, Chapter 8)
        Ix = (2/5) * self.m_s * b**2
        Iy = (1/5) * self.m_s * (a**2 + b**2)
        Iz = Iy
        self.Ig = np.diag([Ix, Iy, Iz])

        # M_S = [  m * I3      -m_s * S(r_s)    m_p * I3
        #        m_s * S(r_s)       Ig             O3
        #         m_p * I3          O3          m_p * I3]
        M_S = np.block([
            [
                self.m*np.eye(3),
                -self.m_s*gnc.linalg.Smtrx(self.r_s),
                self.m_p*np.eye(3)
            ],
            [
                self.m_s*gnc.linalg.Smtrx(self.r_s),
                self.Ig,
                np.zeros((3, 3))
            ],
            [
                self.m_p*np.eye(3),
                np.zeros((3, 3)),
                self.m_p*np.eye(3)
            ]
        ])

        # Weight and buoyancy
        self.W = self.m * gravity
        self.B = self.W

        # Added moment of inertia in roll: A44 = r44 * Ix
        R44 = 0.3
        M_A_44 = R44 * Ix

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
             M_A_44, k_prime*Iy, k_prime*Iy,
             0, 0, 0]
        )

        # System mass matrix
        self.M = M_S + self.M_A

        # Low-speed linear damping matrix parameters
        self.T_surge = 20           # time constant in surge [s]
        self.T_sway = 20            # time constant in sway [s]
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3        # relative damping ratio in roll
        self.zeta_pitch = 0.8       # relative damping ratio in pitch
        self.T_yaw = 1              # time constant in yaw [s]

    def step(self, eta: np.ndarray, r_p: np.ndarray, nu: np.ndarray, v_p: np.ndarray, u_control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Moving mass Remus 100 step method, Rambech et al. (2025) Newton-Euler formulation

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

        alpha = np.arctan2(nu[2], nu[0])             # angle of attack
        U = np.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)  # SOG

        nu_prime = np.concatenate([nu, v_p])

        r_g = (self.m_s*self.r_s + self.m_p*r_p)/self.m

        # Moving mass weight matrix
        # M_P = [   O3         -m_pS(r_p)       O3
        #        m_pS(r_p)   -m_pS(r_p)**2  m_pS(r_p)
        #           O3         -m_pS(r_p)       O3    ]
        M_P = np.block([
            [
                np.zeros((3, 3)),
                -self.m_p*gnc.linalg.Smtrx(r_p),
                np.zeros((3, 3))
            ],
            [
                self.m_p*gnc.linalg.Smtrx(r_p),
                -self.m_p*gnc.linalg.Smtrx(r_p).dot(gnc.linalg.Smtrx(r_p)),
                self.m_p*gnc.linalg.Smtrx(r_p)
            ],
            [
                np.zeros((3, 3)),
                -self.m_p*gnc.linalg.Smtrx(r_p),
                np.zeros((3, 3))
            ]
        ])

        # Total mass matrix
        # M_prime = [ mI3 + A1        -m_pS(r_p)            mpI3
        #            m_pS(r_p)   Ig + A2 - m_pS(r_p)**2   m_pS(r_p)
        #              mpI3           -m_pS(r_p)            mpI3   ]
        M_prime = self.M + M_P
        # Comment inn lines below for linearly independet moving mass
        # M_prime[0:3, 6:9] = np.zeros((3, 3))
        # M_prime[3:6, 6:9] = np.zeros((3, 3))
        # M_prime[6:9, 0:3] = np.zeros((3, 3))
        # M_prime[6:9, 3:6] = np.zeros((3, 3))

        # Invert total mass matrix
        Minv = np.linalg.inv(M_prime)

        # Kinetic energy partial derivatives (Momenta)
        dTdv = (
            M_prime[0:3, 0:3].dot(nu_prime[0:3])
            + M_prime[0:3, 3:6].dot(nu_prime[3:6])
            + M_prime[0:3, 6:9].dot(nu_prime[6:9])
        )

        dTdOmega = (
            M_prime[3:6, 0:3].dot(nu_prime[0:3])
            + M_prime[3:6, 3:6].dot(nu_prime[3:6])
            + M_prime[3:6, 6:9].dot(nu_prime[6:9])
        )

        dTdv_p = (
            M_prime[6:9, 0:3].dot(nu_prime[0:3])
            + M_prime[6:9, 3:6].dot(nu_prime[3:6])
            + M_prime[6:9, 6:9].dot(nu_prime[6:9])
        )

        # C = [[   O3        -S(dT/dv)        O3    ],
        #      [-S(dT/dv)  -S(dT/dOmega)  -S(dT/v_p)],
        #      [   O3       -S(dT/dv_p)       O3    ]]
        C = np.block([
            [
                np.zeros((3, 3)),
                -gnc.linalg.Smtrx(dTdv),
                np.zeros((3, 3))
            ],
            [
                -gnc.linalg.Smtrx(dTdv),
                -gnc.linalg.Smtrx(dTdOmega),
                -gnc.linalg.Smtrx(dTdv_p)
            ],
            [
                np.zeros((3, 3)),
                -gnc.linalg.Smtrx(dTdv_p),
                np.zeros((3, 3))
            ]
        ])

        # Hydrostatics
        f_s = gnc.linalg.Rzyx(eta[3], eta[4], eta[5]).T.dot(
            np.array([0, 0, self.m_s*9.81]))
        f_p = gnc.linalg.Rzyx(eta[3], eta[4], eta[5]).T.dot(
            np.array([0, 0, self.m_p*9.81]))
        f_b = gnc.linalg.Rzyx(eta[3], eta[4], eta[5]).T.dot(
            np.array([0, 0, self.B]))
        g = -np.block([
            f_s + f_p - f_b,
            gnc.linalg.Smtrx(self.r_s).dot(f_s)
            + gnc.linalg.Smtrx(r_p).dot(f_p)
            + gnc.linalg.Smtrx(self.r_b).dot(-f_b),
            f_p
        ])

        # Fictious force to keep m_p
        # from falling out of the vessel
        g_opposite = np.zeros(9)
        g_opposite[6:9] = f_p

        # Hydrodynamic damping
        # Natural frequencies in roll and pitch
        w_roll = np.sqrt(self.W * (r_g[2]-self.r_b[2]) /
                         M_prime[3, 3])
        w_pitch = np.sqrt(self.W * (r_g[2]-self.r_b[2]) /
                          M_prime[4, 4])

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -M_prime[0, 0] / self.T_surge
        Yv = -M_prime[1, 1] / self.T_sway
        Zw = -M_prime[2, 2] / self.T_heave
        Kp = -M_prime[3, 3] * 2 * self.zeta_roll * w_roll
        Mq = -M_prime[4, 4] * 2 * self.zeta_pitch * w_pitch
        Nr = -M_prime[5, 5] / self.T_yaw
        Xu_p = 0
        Yv_p = 0
        Zw_p = 0

        D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr, Xu_p, Yv_p, Zw_p])

        # Vanish at high speed where quadratic drag and lift forces dominates
        D[0, 0] = D[0, 0] * np.exp(-3*U)
        D[1, 1] = D[1, 1] * np.exp(-3*U)

        tau_damp = -np.matmul(D, nu_prime)

        # Nonlinear yaw damping
        tau_damp[5] = tau_damp[5] - 10 * \
            D[5, 5] * abs(nu_prime[5]) * nu_prime[5]

        # State derivatives (with dimension)
        tau_liftdrag = gnc.forceLiftDrag(
            self.d, self.S, self.CD_0, alpha, U
        )
        tau_liftdrag = np.concatenate([tau_liftdrag, [0, 0, 0]])

        # Generalized force vector
        tau_prime = np.zeros(9)
        tau_prime[0] = u_control[0]
        tau_prime[6] = u_control[1]

        # Do not add extra force when at the ends
        # NOTE: Does introduce some chattering
        if r_p[0] >= self.upper_p[0] and u_control[1] > 0:
            tau_prime[6] = 0
        elif r_p[0] <= self.lower_p[0] and u_control[1] < 0:
            tau_prime[6] = 0

        sum_tau = (
            tau_prime
            + tau_damp
            + tau_liftdrag
            - np.matmul(C, nu_prime)
            - g
            - g_opposite
        )

        # Kinetic step
        nu_prime_dot = Minv.dot(sum_tau)
        nu_prime = nu_prime + nu_prime_dot * self.dt

        # Divide system into vessel and moving mass velocities
        nu = nu_prime[0:6]
        v_p = nu_prime[6:9]

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
        eta_dot = gnc.B2N(eta).dot(nu)
        eta = eta + eta_dot * self.dt
        r_p = r_p + r_p_dot * self.dt

        # Saturate moving mass position at the ends
        if r_p[0] >= self.upper_p[0] and u_control[1] > 0:
            r_p[0] = self.upper_p[0]
        elif r_p[0] <= self.lower_p[0] and u_control[1] < 0:
            r_p[0] = self.lower_p[0]

        return eta, r_p, r_p_dot, nu, v_p
