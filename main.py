"""
Remus 100 AUV with Moving Mass

Author: @rambech
"""

# General imports
import matplotlib.pyplot as plt
import numpy as np
import sys

# Spacial imports
import gnc
from src.rambech import Rambech
from src.woolsey import Woolsey

dt = 0.02             # Time step length [s]
time = 500            # Time [s]
steps = int(time/dt)  # Number of simulation steps

vehicle0 = Rambech(dt)
vehicle1 = Woolsey(dt)

eta0 = np.zeros(6)
r_p0 = np.array([0, 0, 0.05])
r_p_dot0 = np.zeros(3)
nu0 = np.zeros(6)
v_p0 = np.zeros(3)
p_p0 = gnc.linalg.Rzyx(eta0[3], eta0[4], eta0[5]).dot(
    r_p0) + eta0[3:6]

eta1 = np.zeros(6)
r_p1 = np.array([0, 0, 0.05])
r_p_dot1 = np.zeros(3)
nu1 = np.zeros(6)
v_p1 = np.zeros(3)
p_p1 = gnc.linalg.Rzyx(eta1[3], eta1[4], eta1[5]).dot(
    r_p1) + eta1[3:6]

# Indicing becomes easier when using time axis 0
eta_data0 = np.zeros((steps, 6))
eta_data0[0] = eta0.copy()
nu_data0 = np.zeros((steps, 6))
nu_data0[0] = nu0.copy()
surge_force_data0 = np.zeros((steps, 1))
pitch_moment_data0 = np.zeros((steps, 1))
r_p_data0 = np.tile(r_p0, (steps, 1))
r_p_dot_data0 = np.zeros((steps, 3))
v_p_data0 = np.zeros((steps, 3))
p_p_data0 = np.tile(p_p0.copy(), (steps, 1))

# Indicing becomes easier when using time axis 0
eta_data1 = np.zeros((steps, 6))
eta_data1[0] = eta1.copy()
nu_data1 = np.zeros((steps, 6))
nu_data1[0] = nu1.copy()
surge_force_data1 = np.zeros((steps, 1))
pitch_moment_data1 = np.zeros((steps, 1))
r_p_data1 = np.tile(r_p1, (steps, 1))
r_p_dot_data1 = np.zeros((steps, 3))
v_p_data1 = np.zeros((steps, 3))
p_p_data1 = np.tile(p_p1.copy(), (steps, 1))

u_d0 = np.zeros(2)
u_d1 = np.zeros(2)
for i in range(1, steps):
    # Progress indicator created by @sibosutd, 12 Oct.
    progress = 100.0*(i+1)/steps
    sys.stdout.write('\r')
    sys.stdout.write("simulating: [{:{}}] {:>3}%"
                     .format('='*int(progress/(100.0/30)),
                             30, int(progress)))
    sys.stdout.flush()

    if eta0[2] < 3:
        u_d0 = np.array([1, 0.5])
    elif eta0[2] > 20:
        u_d0 = np.array([1, -0.5])

    if eta1[2] < 3:
        u_d1 = np.array([1, 0.5])
    elif eta1[2] > 20:
        u_d1 = np.array([1, -0.5])

    # Model step
    eta0, r_p0, r_p_dot0, nu0, v_p0 = vehicle0.step(
        eta0, r_p0, nu0, v_p0, u_d0
    )
    p_p0 = gnc.linalg.Rzyx(eta0[3], eta0[4], eta0[5]).dot(r_p0) + eta0[0:3]

    eta1, r_p1, r_p_dot1, nu1, v_p1 = vehicle1.step(
        eta1, r_p1, nu1, v_p1, u_d1
    )
    p_p1 = gnc.linalg.Rzyx(eta1[3], eta1[4], eta1[5]).dot(r_p1) + eta1[0:3]

    surge_force_data0[i] = u_d0[0]
    pitch_moment_data0[i] = u_d0[1]
    eta_data0[i] = eta0
    r_p_data0[i] = r_p0
    r_p_dot_data0[i] = r_p_dot0
    nu_data0[i] = nu0
    v_p_data0[i] = v_p0
    p_p_data0[i] = p_p0

    surge_force_data1[i] = u_d1[0]
    pitch_moment_data1[i] = u_d1[1]
    eta_data1[i] = eta1
    r_p_data1[i] = r_p1
    r_p_dot_data1[i] = r_p_dot1
    nu_data1[i] = nu1
    v_p_data1[i] = v_p1
    p_p_data1[i] = p_p1

print("\nresults shown in plots")

time_series = np.arange(0, time, dt)

data_precision = 8
surge_force_data0 = np.round(surge_force_data0, data_precision)
pitch_moment_data0 = np.round(pitch_moment_data0, data_precision)
eta_data0 = np.round(eta_data0, data_precision)
r_p_data0 = np.round(r_p_data0, data_precision)
nu_data0 = np.round(nu_data0, data_precision)
v_p_data0 = np.round(v_p_data0, data_precision)
p_p_data0 = np.round(p_p_data0, data_precision)

surge_force_data1 = np.round(surge_force_data1, data_precision)
pitch_moment_data1 = np.round(pitch_moment_data1, data_precision)
eta_data1 = np.round(eta_data1, data_precision)
r_p_data1 = np.round(r_p_data1, data_precision)
nu_data1 = np.round(nu_data1, data_precision)
v_p_data1 = np.round(v_p_data1, data_precision)
p_p_data1 = np.round(p_p_data1, data_precision)

# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

# Plot
# Velocities vs time plot
fig, axes = plt.subplots(3, 1, sharex=True)
axes[1].yaxis.set_inverted(True)
# North
axes[0].plot(time_series, nu_data0[:, 0])
axes[0].plot(time_series, nu_data1[:, 0], linestyle="dashed")
# Down
axes[1].plot(time_series, nu_data0[:, 2])
axes[1].plot(time_series, nu_data1[:, 2], linestyle="dashed")
# Pitch
axes[2].plot(time_series, nu_data0[:, 4]*180/np.pi)
axes[2].plot(time_series, nu_data1[:, 4]*180/np.pi, linestyle="dashed")
# Labels and titles
axes[0].set_ylabel(r"$u\ [\text{m/s}]$")
axes[1].set_ylabel(r"$w\ [\text{m/s}]$")
axes[2].set_ylabel(r"$q\ [^\circ\text{/s}]$")
for axis in axes:
    axis.grid()
    axis.set_xlim(97, 112)

# Moving mass plot
fig1, axes1 = plt.subplots(2, 1, sharex=True)
axes1[0].plot(time_series, r_p_data0[:, 0])
axes1[0].plot(time_series, r_p_data1[:, 0], linestyle="dashed")
axes1[1].plot(time_series, r_p_dot_data0[:, 0])
axes1[1].plot(time_series, r_p_dot_data1[:, 0], linestyle="dashed")
axes1[0].set_ylabel(r"$x_p\ [\text{m}]$")
axes1[1].set_ylabel(r"$\dot{x}_p\ [\text{m/s}]$")
axes1[-1].set_xlabel(r"Time $[\text{s}]$")
for axis in axes1:
    axis.grid()
    axis.set_xlim(97, 112)

# Vehicle pose vs time
fig2, axes2 = plt.subplots(2, 1, sharex=True)
axes2[1].yaxis.set_inverted(True)
# Down
axes2[0].plot(time_series, eta_data0[:, 2])
axes2[0].plot(time_series, eta_data1[:, 2],
              linestyle="dashed")
# Pitch
axes2[1].plot(time_series, eta_data0[:, 4]*180/np.pi)
axes2[1].plot(time_series, eta_data1[:, 4]*180/np.pi, linestyle="dashed")
# Labels and titles
axes2[0].set_ylabel(r"Down $[\text{m}]$")
axes2[1].set_ylabel(r"Pitch $[^\circ]$")
axes2[-1].set_xlabel(r"Time $[\text{s}]$")
axes2[0].yaxis.set_inverted(True)
for axis in axes2:
    axis.grid()
    axis.set_xlim(97, 112)

lw = 2
# Master figure
fig3, axes3 = plt.subplots(9, 1, sharex=True, figsize=(5.25, 8))
# Down
axes3[0].plot(time_series, eta_data0[:, 2],
              lw=lw, label="Rambech et al. 2025")
axes3[0].plot(time_series, eta_data1[:, 2], lw=lw,
              linestyle="dashed", label="Woolsey and Leonard 2002")
axes3[0].set_yticks([0, 10, 20], [0, 10, 20])
axes3[0].set_ylim(-2.5, 23)
# Pitch
axes3[1].plot(time_series, eta_data0[:, 4]*180/np.pi, lw=lw,)
axes3[1].plot(time_series, eta_data1[:, 4]*180 /
              np.pi, lw=2, linestyle="dashed")
# x_p
axes3[2].plot(time_series, r_p_data0[:, 0], lw=lw)
axes3[2].plot(time_series, r_p_data1[:, 0], lw=lw, linestyle="dashed")
axes3[2].set_ylim(-0.07, 0.07)
# x_p_dot
axes3[3].plot(time_series, r_p_dot_data0[:, 0], lw=lw)
axes3[3].plot(time_series, r_p_dot_data1[:, 0], lw=lw, linestyle="dashed")
axes3[3].set_yticks([-0.25, 0, 0.17], [-0.25, 0, 0.17])
# Surge
axes3[4].plot(time_series, nu_data0[:, 0], lw=lw,)
axes3[4].plot(time_series, nu_data1[:, 0], lw=lw, linestyle="dashed")
axes3[4].set_yticks([0, 0.2, 0.35], [0, 0.2, 0.35])
# Heave
axes3[5].plot(time_series, nu_data0[:, 2], lw=lw)
axes3[5].plot(time_series, nu_data1[:, 2], lw=lw, linestyle="dashed")
# Pitch rate
axes3[6].plot(time_series, nu_data0[:, 4]*180/np.pi, lw=lw)
axes3[6].plot(time_series, nu_data1[:, 4]*180 /
              np.pi, lw=lw, linestyle="dashed")
# Surge force
axes3[7].plot(time_series, surge_force_data0, lw=lw)
axes3[7].plot(time_series, surge_force_data1, lw=lw, linestyle="dashed")
axes3[7].set_ylim(-0.2, 1.2)
# Moving mass force
axes3[8].plot(time_series, pitch_moment_data0, lw=lw)
axes3[8].plot(time_series, pitch_moment_data1, lw=lw, linestyle="dashed")
axes3[8].set_ylim(-0.7, 0.7)
# Labels, legends and titles
axes3[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0.)
axes3[0].set_ylabel(r"$z\ [\text{m}]$")
axes3[1].set_ylabel(r"$\theta\ [^\circ]$")
axes3[2].set_ylabel(r"$x_p\ [\text{m}]$")
axes3[3].set_ylabel(r"$\dot{x}_p\ [\text{m/s}]$")
axes3[4].set_ylabel(r"$u\ [\text{m/s}]$")
axes3[5].set_ylabel(r"$w\ [\text{m/s}]$")
axes3[6].set_ylabel(r"$q\ [^\circ\text{/s}]$")
axes3[7].set_ylabel(r"$\tau_X$")
axes3[8].set_ylabel(r"$\tau_{X_p}$")
axes3[-1].set_xlabel(r"Time $[\text{s}]$")
axes3[0].yaxis.set_inverted(True)
for axis in axes3:
    axis.grid(which="both")
fig3.align_ylabels()

# Master figure close up
fig4, axes4 = plt.subplots(9, 1, sharex=True, figsize=(5.25, 8))
# Down
axes4[0].plot(time_series, eta_data0[:, 2],
              lw=lw, label="Rambech et al. 2025")
axes4[0].plot(time_series, eta_data1[:, 2], lw=lw,
              linestyle="dashed", label="Woolsey and Leonard 2002")
axes4[0].set_ylim(-0.2, 1.2)
# Pitch
axes4[1].plot(time_series, eta_data0[:, 4]*180/np.pi, lw=lw)
axes4[1].plot(time_series, eta_data1[:, 4]*180 /
              np.pi, lw=lw, linestyle="dashed")
axes4[1].set_ylim(-60, 5)
# x_p
axes4[2].plot(time_series, r_p_data0[:, 0], lw=lw)
axes4[2].plot(time_series, r_p_data1[:, 0], lw=lw, linestyle="dashed")
axes4[2].set_ylim(-0.01, 0.06)
# x_p_dot
axes4[3].plot(time_series, r_p_dot_data0[:, 0], lw=lw)
axes4[3].plot(time_series, r_p_dot_data1[:, 0], lw=lw, linestyle="dashed")
axes4[3].set_ylim(-0.02, 0.12)
# Surge
axes4[4].plot(time_series, nu_data0[:, 0], lw=lw)
axes4[4].plot(time_series, nu_data1[:, 0], lw=lw, linestyle="dashed")
axes4[4].set_ylim(-0.05, 0.3)
# Heave
axes4[5].plot(time_series, nu_data0[:, 2], lw=lw)
axes4[5].plot(time_series, nu_data1[:, 2], lw=lw, linestyle="dashed")
axes4[5].set_ylim(-0.05, 0.05)
# Pitch rate
axes4[6].plot(time_series, nu_data0[:, 4]*180/np.pi, lw=lw)
axes4[6].plot(time_series, nu_data1[:, 4]*180 /
              np.pi, lw=lw, linestyle="dashed")
axes4[6].set_ylim(-30, 30)
# Surge force
axes4[7].plot(time_series, surge_force_data0, lw=lw)
axes4[7].plot(time_series, surge_force_data1, lw=lw, linestyle="dashed")
axes4[7].set_ylim(-0.2, 1.2)
# Moving mass force
axes4[8].plot(time_series, pitch_moment_data0, lw=lw)
axes4[8].plot(time_series, pitch_moment_data1, lw=lw, linestyle="dashed")
axes4[8].set_ylim(-0.1, 0.6)
# Labels, legends and titles
axes4[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0.)
axes4[0].set_ylabel(r"$z\ [\text{m}]$")
axes4[1].set_ylabel(r"$\theta\ [^\circ]$")
axes4[2].set_ylabel(r"$x_p\ [\text{m}]$")
axes4[3].set_ylabel(r"$\dot{x}_p\ [\text{m/s}]$")
axes4[4].set_ylabel(r"$u\ [\text{m/s}]$")
axes4[5].set_ylabel(r"$w\ [\text{m/s}]$")
axes4[6].set_ylabel(r"$q\ [^\circ\text{/s}]$")
axes4[7].set_ylabel(r"$\tau_X$")
axes4[8].set_ylabel(r"$\tau_{X_p}$")
axes4[-1].set_xlabel(r"Time $[\text{s}]$")
axes4[0].yaxis.set_inverted(True)
for axis in axes4:
    axis.grid(which="both")
    axis.set_xlim(-0.5, 12)
fig4.align_ylabels()
plt.show()
