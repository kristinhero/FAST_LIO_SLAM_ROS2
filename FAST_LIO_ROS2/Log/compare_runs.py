import os
import numpy as np
import matplotlib.pyplot as plt

_base    = os.path.dirname(os.path.abspath(__file__))
_dir_bad = os.path.join(_base, 'Log_square_tank_slow_bad')
_dir_suc = os.path.join(_base)


def load_mat(d, name):
    return np.loadtxt(os.path.join(d, name))

def load_pos_log(d):
    pos = np.loadtxt(os.path.join(d, 'pos_log.txt'))
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    return pos


a_out_b = load_mat(_dir_bad, 'mat_out.txt')
a_pre_b = load_mat(_dir_bad, 'mat_pre.txt')
time_b  = a_pre_b[:, 0]

a_out_s = load_mat(_dir_suc, 'mat_out.txt')
a_pre_s = load_mat(_dir_suc, 'mat_pre.txt')
time_s  = a_pre_s[:, 0]

# --- XY trajectory ---
fig, ax = plt.subplots()
fig.suptitle('Estimated Trajectory (XY)')
ax.plot(a_out_b[:, 4], a_out_b[:, 5], '-',  color='tab:red',  label='bad')
ax.plot(a_out_s[:, 4], a_out_s[:, 5], '--', color='tab:blue', label='success')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()
ax.legend()

# --- Accelerometer bias ---
colors = ['tab:blue', 'tab:orange', 'tab:green']
xyz = ['x', 'y', 'z']
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Accelerometer Bias Comparison (bad=solid, success=dashed)')
for i in range(3):
    axs[i].plot(time_b, a_out_b[:, (i+1)+6*3], '-',  color=colors[i], label=f'bad-{xyz[i]}')
    axs[i].plot(time_s, a_out_s[:, (i+1)+6*3], '--', color=colors[i], label=f'success-{xyz[i]}', alpha=0.7)
    axs[i].set_ylabel('Bias [m/s^2]')
    axs[i].grid()
    axs[i].legend()
axs[-1].set_xlabel('Time [s]')
plt.tight_layout()

# --- EKF information matrix ---
pos_b = load_pos_log(_dir_bad)
pos_s = load_pos_log(_dir_suc)

def compute_eig(pos):
    info_flat = pos[:, 64:100]  # (N, 36)
    info_mats = info_flat.reshape(-1, 6, 6)
    N = len(pos)
    eig_vals = np.zeros((N, 6))
    eig_vecs = np.zeros((N, 6, 6))
    for k in range(N):
        vals, vecs = np.linalg.eigh(info_mats[k])
        eig_vals[k] = vals
        eig_vecs[k] = vecs
    cond = np.where(eig_vals[:, 5] > 1e-10, eig_vals[:, 0] / eig_vals[:, 5], 0.0)
    return eig_vals, eig_vecs, cond


if pos_b.shape[1] >= 61 + 3 + 36 and pos_s.shape[1] >= 61 + 3 + 36:
    t_b = pos_b[:, 0];  t_s = pos_s[:, 0]
    eig_vals_b, eig_vecs_b, cond_b = compute_eig(pos_b)
    eig_vals_s, eig_vecs_s, cond_s = compute_eig(pos_s)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EKF Information Matrix Analysis (bad=solid red, success=dashed blue)')

    axs[0, 0].plot(t_b, pos_b[:, 61], '-',  color='tab:red',  label='bad')
    axs[0, 0].plot(t_s, pos_s[:, 61], '--', color='tab:blue', label='success', alpha=0.7)
    axs[0, 0].set_title('Effective Points')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(t_b, pos_b[:, 62], '-',  color='tab:red',    label='mean res bad')
    axs[0, 1].plot(t_b, pos_b[:, 63], '-',  color='tab:orange', label='cost bad')
    axs[0, 1].plot(t_s, pos_s[:, 62], '--', color='tab:blue',   label='mean res success', alpha=0.7)
    axs[0, 1].plot(t_s, pos_s[:, 63], '--', color='tab:green',  label='cost success',     alpha=0.7)
    axs[0, 1].set_title('Residuals / Cost')
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].legend()
    axs[0, 1].grid()

    eig_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    for i in range(6):
        axs[1, 0].plot(t_b, eig_vals_b[:, i], '-',  color=eig_colors[i], label=f'eig_{i+1} bad')
        axs[1, 0].plot(t_s, eig_vals_s[:, i], '--', color=eig_colors[i], alpha=0.5, label=f'eig_{i+1} suc')
    axs[1, 0].set_title('Info Matrix Eigenvalues (ascending)')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_yscale('log')
    axs[1, 0].legend(fontsize=7)
    axs[1, 0].grid()

    axs[1, 1].plot(t_b, cond_b, '-',  color='tab:red',  label='bad')
    axs[1, 1].plot(t_s, cond_s, '--', color='tab:blue', label='success', alpha=0.7)
    axs[1, 1].set_title('Condition Number (eig_min / eig_max)')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()

    # Weakest eigenvector components: bad vs success
    pose_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z']
    fig5, axs5 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig5.suptitle('Weakest Eigenvector Components (eig_1 direction)')
    for i in range(6):
        axs5[0].plot(t_b, np.abs(eig_vecs_b[:, i, 0]), label=pose_labels[i])
        axs5[1].plot(t_s, np.abs(eig_vecs_s[:, i, 0]), label=pose_labels[i])
    axs5[0].set_title('bad');    axs5[0].set_ylabel('|component|'); axs5[0].legend(); axs5[0].grid()
    axs5[1].set_title('success'); axs5[1].set_ylabel('|component|'); axs5[1].legend(); axs5[1].grid()
    axs5[1].set_xlabel('Time [s]')
    plt.tight_layout()
else:
    print('pos_log.txt missing info matrix columns in one or both runs.')

plt.show()
