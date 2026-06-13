# import matplotlib
# matplotlib.use('Agg')
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_style, fig_size, save, mark

# --- CLI / styling ---
parser = argparse.ArgumentParser(
    description='Plot FAST-LIO logs (interactive, or save figures to PDF).')
parser.add_argument('--out-dir', default=None,
                    help='If set, save mapped figures to <out-dir>/<stem>.pdf '
                         'and suppress plt.show(). If unset, behavior is interactive.')
parser.add_argument('--no-pre', action='store_true',
                    help='Suppress dashed pre-update traces on state plots.')
parser.add_argument('--log-dir', default='Log_no_motion_deskew/curved_tank_fast',
                    help='Dataset directory (absolute, or relative to this script).')
parser.add_argument('--heatmap-t', type=float, default=None,
                    help='Timestamp [s] for the eigenvector heatmap. Default: per-dataset lookup, else 688.')
parser.add_argument('--heatmap-thresh', type=float, default=None,
                    help='Eigenvalue threshold annotated in the heatmap title. Default: per-dataset lookup, else 300.')
args = parser.parse_args()

apply_style()

# Display name per dataset (used as the Registration figure title and heatmap prefix).
DATASET_NAME = {
    'Log_square_tank_slow': 'Slow-A',
    'Log_square_tank_fast': 'Fast-A',
    'Log_curved_tank_slow': 'Slow-B',
    'Log_curved_tank_fast': 'Fast-B',
}

# Per-dataset eigenvector-heatmap scan used when --heatmap-* are not given.
# Keyed by dataset directory name -> (timestamp [s], eigenvalue threshold).
HEATMAP_BY_DATASET = {
    'Log_curved_tank_fast': (687.4, 300.0),   # degenerate scan: 4 weak dirs (X, Y, Z + roll)
    'Log_curved_tank_slow': (537.7, 2000.0),  # representative scan: 3 weak dirs (X, Y, Z)
}
HEATMAP_DEFAULT = (688.0, 300.0)

# Figures saved at their full canvas width (no tight-crop) so they scale
# identically when included at the same width in LaTeX (e.g. 0.5*textwidth).
FIXED_WIDTH_STEMS = {'Trajectory', 'Trajectory_3D', 'Heatmap_eigen'}

_dir = (args.log_dir if os.path.isabs(args.log_dir)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.log_dir))

# Resolve heatmap scan: explicit flag > per-dataset lookup > global default.
_heat_t_def, _heat_thr_def = HEATMAP_BY_DATASET.get(os.path.basename(os.path.normpath(_dir)), HEATMAP_DEFAULT)
heat_t = args.heatmap_t if args.heatmap_t is not None else _heat_t_def
heat_thresh = args.heatmap_thresh if args.heatmap_thresh is not None else _heat_thr_def
ds_name = DATASET_NAME.get(os.path.basename(os.path.normpath(_dir)), 'Registration Analysis')


#######for ikfom
lab_pre = ['', 'pre-x', 'pre-y', 'pre-z']
lab_out = ['', 'out-x', 'out-y', 'out-z']
a_pre=np.loadtxt(os.path.join(_dir, 'mat_pre.txt'))
a_out=np.loadtxt(os.path.join(_dir, 'mat_out.txt'))
time=a_pre[:,0]

# --- Attitude (j=0) ---
att_labels = ['roll [deg]', 'pitch [deg]', 'yaw [deg]']
pose_colors = ['tab:blue', 'tab:green', 'tab:orange']   # shared by Attitude, Position, Velocity
fig, axes = plt.subplots(3, 1, figsize=fig_size(1.0, 0.7), sharex=True)
fig.suptitle('Attitude')
mark(fig, 'Attitude')
for i in range(3):
    if not args.no_pre:
        axes[i].plot(time, a_pre[:, i+1], '--', color=pose_colors[i], label='pre')
    axes[i].plot(time, a_out[:, i+1], '-', color=pose_colors[i], label='out')
    axes[i].set_ylabel(att_labels[i])
    axes[i].grid()
    if not args.no_pre:
        axes[i].legend(fontsize=8, loc='center right')
axes[-1].set_xlabel('Time [s]')
fig.align_ylabels(axes)
plt.tight_layout(h_pad=0.3)

# --- Position (translation, j=1) ---
trans_labels = ['x [m]', 'y [m]', 'z [m]']
fig, axes = plt.subplots(3, 1, figsize=fig_size(1.0, 0.7), sharex=True)
fig.suptitle('Position')
mark(fig, 'Position')
for i in range(3):
    if not args.no_pre:
        axes[i].plot(time, a_pre[:, i+4], '--', color=pose_colors[i], label='pre')
    axes[i].plot(time, a_out[:, i+4], '-', color=pose_colors[i], label='out')
    axes[i].set_ylabel(trans_labels[i])
    axes[i].grid()
    if not args.no_pre:
        axes[i].legend(fontsize=8, loc='center right')
axes[-1].set_xlabel('Time [s]')
fig.align_ylabels(axes)
plt.tight_layout(h_pad=0.3)

# --- XY trajectory ---
fig, ax = plt.subplots(figsize=fig_size(0.5, 1.0))   # author at 0.5*textwidth -> no LaTeX rescale
fig.suptitle('Estimated Trajectory (XY)')
mark(fig, 'Trajectory')
ax.plot(a_out[:, 4], a_out[:, 5], '-', label='out')
if not args.no_pre:
    ax.plot(a_pre[:, 4], a_pre[:, 5], '--', label='pre')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()
if not args.no_pre:
    ax.legend()

# --- 3D trajectory ---
fig3d = plt.figure(figsize=fig_size(0.5, 0.9), layout='none')  # 0.5*textwidth; manual margins (constrained_layout clips 3D)
ax3d = fig3d.add_subplot(111, projection='3d')
fig3d.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.90)
fig3d.suptitle('Estimated Trajectory (3D)')
mark(fig3d, 'Trajectory_3D')
ax3d.plot(a_out[:, 4], a_out[:, 5], a_out[:, 6], '-', label='out', color='m', lw=0.8)
if not args.no_pre:
    ax3d.plot(a_pre[:, 4], a_pre[:, 5], a_pre[:, 6], '--', label='pre', color='y', lw=0.8)
ax3d.set_xlabel('x [m]')
ax3d.set_ylabel('y [m]')
ax3d.set_zlabel('z [m]')
if not args.no_pre:
    ax3d.legend()
ax3d.grid()
pad = 0.5
ax3d.set_xlim(a_out[:, 4].min() - pad, a_out[:, 4].max() + pad)
ax3d.set_ylim(a_out[:, 5].min() - pad, a_out[:, 5].max() + pad)
ax3d.set_zlim(a_out[:, 6].min() - pad, a_out[:, 6].max() + pad)

# --- Velocity (j=4): all three components in one panel (style matches the Bias plots) ---
vel_labels = ['$v_x$', '$v_y$', '$v_z$']
fig, ax = plt.subplots(figsize=fig_size(1.0, 0.32))
fig.suptitle('Velocity')
mark(fig, 'Velocity')
for i in range(3):
    if not args.no_pre:
        ax.plot(time, a_pre[:, i+13], '--', color=pose_colors[i], label=f'{vel_labels[i]} (pre)')
    ax.plot(time, a_out[:, i+13], '-', color=pose_colors[i], label=vel_labels[i])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Velocity [m/s]')
ax.grid()
ax.legend(fontsize=8, loc='center right')

# --- bg (j=5) ---
fig, ax = plt.subplots(figsize=fig_size(1.0, 0.32))
fig.suptitle('Bias of Gyroscope')
mark(fig, 'Bias_g')
for i in range(1, 4):
    if not args.no_pre:
        ax.plot(time, a_pre[:, i+5*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+5*3], '-', label=(lab_out[i][4:] if args.no_pre else lab_out[i]))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Bias [rad/s]')
ax.grid()
ax.legend()

# --- ba (j=6) ---
fig, ax = plt.subplots(figsize=fig_size(1.0, 0.32))
fig.suptitle('Bias of Accelerometer')
mark(fig, 'Bias_a')
for i in range(1, 4):
    if not args.no_pre:
        ax.plot(time, a_pre[:, i+6*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+6*3], '-', label=(lab_out[i][4:] if args.no_pre else lab_out[i]))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Bias [m/s$^2$]')
ax.grid()
ax.legend()

# --- Gravity (j=7) ---
fig, ax = plt.subplots(figsize=fig_size(1.0, 0.32))
fig.suptitle('Estimated Gravity')
mark(fig, 'Gravity')
for i in range(1, 4):
    if not args.no_pre:
        ax.plot(time, a_pre[:, i+7*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+7*3], '-', label=(lab_out[i][4:] if args.no_pre else lab_out[i]))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Gravity [m/s$^2$]')
ax.grid()
ax.legend()
#######for ikfom#######

# ### Calculate and plot state differences (out - pre)
# state_cfg = {
#     0: ('Attitude',     ['Δroll [deg]',  'Δpitch [deg]', 'Δyaw [deg]'],   True),
#     1: ('Translation',  ['Δx [m]',       'Δy [m]',       'Δz [m]'],       False),
#     4: ('Velocity',     ['Δvx [m/s]',    'Δvy [m/s]',    'Δvz [m/s]'],    False),
#     5: ('Gyro bias',    ['Δbg_x [rad/s]','Δbg_y [rad/s]','Δbg_z [rad/s]'],False),
#     6: ('Acc bias',     ['Δba_x [m/s²]', 'Δba_y [m/s²]', 'Δba_z [m/s²]'],False),
#     7: ('Gravity',      ['Δgx [m/s²]',   'Δgy [m/s²]',   'Δgz [m/s²]'],  False),
# }
# for j, (name, ylabels, wrap) in state_cfg.items():
#     fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
#     fig.suptitle(f'State difference (out − pre): {name}')
#     for i in range(3):
#         diff = a_out[:, i+1+j*3] - a_pre[:, i+1+j*3]
#         if wrap:
#             diff = (diff + 180) % 360 - 180
#         axes[i].plot(time, diff, '-', color=f'C{i}', label=ylabels[i])
#         axes[i].set_ylabel(ylabels[i])
#         axes[i].legend(fontsize=8, loc='center right')
#         axes[i].grid()
#     axes[-1].set_xlabel('Time [s]')
#     plt.tight_layout()


# --- IEKF correction magnitude (out - pre) ---
fig, axes = plt.subplots(2, 1, figsize=fig_size(1.0, 0.45), sharex=True)
fig.suptitle('IEKF Correction Magnitude')
mark(fig, 'Correction')

dp = a_out[:, 4:7] - a_pre[:, 4:7]
norm_p = np.linalg.norm(dp, axis=1)

try:
    from scipy.spatial.transform import Rotation as R
    # Euler cols 1..3 are roll, pitch, yaw in degrees, XYZ intrinsic —
    # same convention as the Attitude block above.
    R_pre = R.from_euler('xyz', a_pre[:, 1:4], degrees=True)
    R_out = R.from_euler('xyz', a_out[:, 1:4], degrees=True)
    dR = R_pre.inv() * R_out
    angle_deg = np.degrees(np.linalg.norm(dR.as_rotvec(), axis=1))
except ImportError:
    # Fallback: per-axis Euler difference magnitude. Approximation valid
    # for small corrections only (ignores rotation non-commutativity).
    diff = a_out[:, 1:4] - a_pre[:, 1:4]
    diff = (diff + 180) % 360 - 180        # wrap each axis to [-180, 180]
    angle_deg = np.linalg.norm(diff, axis=1)

axes[0].plot(time, norm_p, '-')
axes[0].set_ylabel('Δp [m]')
axes[0].grid()
axes[1].plot(time, angle_deg, '-')
axes[1].set_ylabel('ΔΘ [deg]')
axes[1].set_xlabel('Time [s]')
axes[1].grid()
fig.align_ylabels(axes)
plt.tight_layout()

### Draw IMU data
fig, axs = plt.subplots(2)
imu=np.loadtxt(os.path.join(_dir, 'imu.txt'))
time=imu[:,0]
axs[0].set_title('Gyroscope')
axs[1].set_title('Accelerometer')
lab_1 = ['gyr-x', 'gyr-y', 'gyr-z']
lab_2 = ['acc-x', 'acc-y', 'acc-z']
for i in range(3):
    axs[0].plot(time, imu[:,i+1],'-', label=lab_1[i])
    axs[1].plot(time, imu[:,i+4],'-', label=lab_2[i])
for i in range(2):
    axs[i].grid()
    axs[i].legend()
plt.tight_layout(h_pad=2.0)

# #### Draw time calculation
# plt.figure(3)
# fig = plt.figure()
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 12,
# }
# c="red"
# a_out1=np.loadtxt('Log/mat_out_time_indoor1.txt')
# a_out2=np.loadtxt('Log/mat_out_time_indoor2.txt')
# a_out3=np.loadtxt('Log/mat_out_time_outdoor.txt')
# # n = a_out[:,1].size
# # time_mean = a_out[:,1].mean()
# # time_se   = a_out[:,1].std() / np.sqrt(n)
# # time_err  = a_out[:,1] - time_mean
# # feat_mean = a_out[:,2].mean()
# # feat_err  = a_out[:,2] - feat_mean
# # feat_se   = a_out[:,2].std() / np.sqrt(n)
# ax1 = fig.add_subplot(111)
# ax1.set_ylabel('Effective Feature Numbers',font1)
# ax1.boxplot(a_out1[:,2], showfliers=False, positions=[0.9])
# ax1.boxplot(a_out2[:,2], showfliers=False, positions=[1.9])
# ax1.boxplot(a_out3[:,2], showfliers=False, positions=[2.9])
# ax1.set_ylim([0, 3000])

# ax2 = ax1.twinx()
# ax2.spines['right'].set_color('red')
# ax2.set_ylabel('Compute Time (ms)',font1)
# ax2.yaxis.label.set_color('red')
# ax2.tick_params(axis='y', colors='red')
# ax2.boxplot(a_out1[:,1]*1000, showfliers=False, positions=[1.1],boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c))
# ax2.boxplot(a_out2[:,1]*1000, showfliers=False, positions=[2.1],boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c))
# ax2.boxplot(a_out3[:,1]*1000, showfliers=False, positions=[3.1],boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c))
# ax2.set_xlim([0.5, 3.5])
# ax2.set_ylim([0, 100])

# plt.xticks([1,2,3], ('Outdoor Scene', 'Indoor Scene 1', 'Indoor Scene 2'))
# # print(time_se)
# # print(a_out3[:,2])
# plt.grid()
# plt.savefig("time.pdf", dpi=1200)

# Plot covariance diagonals from pos_log.txt if available
try:
    pos = np.loadtxt(os.path.join(_dir, 'pos_log.txt'))
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    # pos columns: 0..24 state fields, then covariance, then 3 scalars, then H^T*H.
    # Two on-disk covariance layouts are auto-detected from the column count:
    #   full pose covariance: cols 25-60 = P (6x6 row-major)      -> N_P=36, 100 cols
    #   state-diagonal cov:   cols 25-47 = diag (pos,rot,...,grav) -> N_P=23,  87 cols
    # In both, the first 6 diagonal entries are the pose block (pos x/y/z, rot x/y/z).
    COL_P    = 25
    HMAT_SZ  = 36          # H^T*H is always a 6x6
    N_SCALAR = 3           # n_pts, mean_res, cost
    N_P      = pos.shape[1] - COL_P - N_SCALAR - HMAT_SZ   # 36 (full 6x6) or 23 (diagonal)
    COL_INFO = COL_P + N_P            # n_pts, mean_res, cost
    COL_HMAT = COL_INFO + N_SCALAR    # H^T*H (6x6 row-major)
    full_cov = (N_P == 36)            # full pose covariance matrix vs logged state diagonal

    time_pos = pos[:, 0]

    if N_P >= 6:
        t_cov = time_pos
        # 6 pose-block diagonal variances (pos x/y/z, rot x/y/z) for either layout
        if full_cov:
            cov   = pos[:, COL_P:COL_P + N_P].reshape(-1, 6, 6)
            diag  = np.array([cov[:, i, i] for i in range(6)])   # (6, N)
        else:
            diag  = pos[:, COL_P:COL_P + 6].T                    # first 6 diag entries = pose
        cov_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        fig2, axes_cov = plt.subplots(2, 3, figsize=fig_size(1.0, 0.55))
        axes_cov = axes_cov.ravel()
        for i in range(6):
            ax = axes_cov[i]
            ax.plot(t_cov, diag[i])
            ax.set_ylabel(cov_labels[i], labelpad=1)   # pull label tight against the subplot
            ax.tick_params(labelsize=7)                # narrower tick labels -> tighter columns
            ax.grid()
        fig2.suptitle('Pose Covariance Diagonal')
        mark(fig2, 'Covariance')
        plt.tight_layout(h_pad=0.1, w_pad=0.0)
    else:
        print('Log/pos_log.txt: unexpected column count %d (need >=87).' % pos.shape[1])

    # --- Information matrix analysis ---
    # COL_INFO=n_pts, +1=mean_res, +2=cost; H^T*H (6x6) at COL_HMAT (offsets auto-detected above)
    if N_P >= 6:
        t_info   = time_pos
        n_pts    = pos[:, COL_INFO]
        mean_res = pos[:, COL_INFO + 1]
        cost     = pos[:, COL_INFO + 2]
        info_flat = pos[:, COL_HMAT:COL_HMAT + 36]  # (N, 36)
        info_mats = info_flat.reshape(-1, 6, 6)

        # Compute eigenvalues and eigenvectors per timestep
        eig_vals = np.zeros((len(t_info), 6))
        eig_vecs = np.zeros((len(t_info), 6, 6))  # columns are eigenvectors
        for k in range(len(t_info)):
            vals, vecs = np.linalg.eigh(info_mats[k])  # ascending order
            eig_vals[k] = vals
            eig_vecs[k] = vecs

        cond = np.where(eig_vals[:, 5] > 1e-10, eig_vals[:, 0] / eig_vals[:, 5], 0.0)

        # Dominant state label for the weakest eigenvector (largest component)
        pose_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z']
        weakest_vec = eig_vecs[:, :, 0]  # eigenvector for eig_1 (minimum)
        dominant_idx = np.argmax(np.abs(weakest_vec), axis=1)

        fig3, axs = plt.subplots(2, 2, figsize=fig_size(1.0, 0.85))
        fig3.suptitle(ds_name)
        mark(fig3, 'IEKF')

        axs[0, 0].plot(t_info, n_pts)
        axs[0, 0].axhline(n_pts.mean(), color='r', ls='--', lw=1, label='mean %.0f' % n_pts.mean())
        axs[0, 0].set_title('Effective Points')
        axs[0, 0].legend(fontsize=8)
        axs[0, 0].grid()

        # axs[0, 1].plot(t_info, mean_res, label='mean residual [m]')
        axs[0, 1].plot(t_info, cost)
        axs[0, 1].axhline(cost.mean(), color='r', ls='--', lw=1, label='mean %.3g' % cost.mean())
        axs[0, 1].set_title('Residuals')
        axs[0, 1].legend()
        axs[0, 1].grid()

        eig_labels = [f'$\\lambda_{{{i+1}}}$' for i in range(6)]
        for i in range(6):
            axs[1, 0].plot(t_info, eig_vals[:, i], label=eig_labels[i])
        axs[1, 0].set_title('Eigenvalues')
        axs[1, 0].set_yscale('log')
        axs[1, 0].legend(fontsize=6, ncol=3, framealpha=0.5, loc='best')
        axs[1, 0].grid()

        axs[1, 1].plot(t_info, cond)
        axs[1, 1].axhline(cond.mean(), color='r', ls='--', lw=1, label='mean %.3g' % cond.mean())
        axs[1, 1].set_title('Inverse Condition Number')
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].grid()

        plt.tight_layout()

        # Weakest eigenvector components over time
        fig4, axs4 = plt.subplots(figsize=(12, 4))
        fig4.suptitle('Weakest Eigenvector Components (eig_1 direction)')
        for i in range(6):
            axs4.plot(t_info, np.abs(weakest_vec[:, i]), label=pose_labels[i])
        axs4.set_xlabel('Time [s]')
        axs4.set_ylabel('|component|')
        axs4.legend()
        axs4.grid()
        plt.tight_layout()

        # --- Eigenvector heatmaps: X, Y, Z — sorted by peak time ---
        # H^T*H cols: 0=pos_x, 1=pos_y, 2=pos_z, 3=rot_x, 4=rot_y, 5=rot_z
        # Reorder to rot-first (paper style): [3,4,5,0,1,2] → Rx,Ry,Rz,X,Y,Z
        reorder    = [3, 4, 5, 0, 1, 2]
        col_labels = ['Rx', 'Ry', 'Rz', 'X', 'Y', 'Z']

        from matplotlib.colors import PowerNorm
        gamma = 0.35   # <1 spreads low values; raise toward 1.0 for more linear
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=1)

        # (label, weakest_vec column index)
        panel_defs = [('X', 0), ('Y', 1), ('Z', 2)]

        # Compute peak time index for each panel, then sort by time
        panels = sorted(
            [(label, col, int(np.argmax(np.abs(weakest_vec[:, col]))))
             for label, col in panel_defs],
            key=lambda x: t_info[x[2]]
        )

        # # 4 columns: 3 heatmaps + 1 narrow colorbar column
        # fig_h, axes_h = plt.subplots(1, 4, figsize=(12, 5),
        #                              gridspec_kw={'width_ratios': [4, 4, 4, 0.4]})
        # fig_h.suptitle('Eigenvector heatmaps at peak X / Y / Z dominance\n'
        #                '(sorted by time — rows = v1..v6 ascending eigenvalue, cols = Rx Ry Rz X Y Z)', fontsize=9)
        #
        # for panel, (label, orig_col, peak_idx) in enumerate(panels):
        #     mat = info_mats[peak_idx][:, reorder][reorder, :]
        #     vals, vecs = np.linalg.eigh(mat)
        #     data = np.abs(vecs.T)          # row i = eigenvector i (ascending)
        #
        #     ax = axes_h[panel]
        #     im = ax.imshow(data, cmap='Greys_r', norm=norm, aspect='auto')
        #
        #     # red separator at largest eigenvalue gap
        #     if vals[-1] > 1e-10:
        #         ratios = np.diff(np.log10(np.clip(vals, 1e-10, None)))
        #         gap = int(np.argmax(ratios))
        #         ax.axhline(gap + 0.5, color='red', lw=1.2)
        #
        #     ax.set_title(f'{label} dominant  t={t_info[peak_idx]:.2f}s', fontsize=8)
        #     ax.set_xticks(range(6))
        #     ax.set_xticklabels(col_labels, fontsize=7, rotation=45)
        #     ax.set_yticks(range(6))
        #     if panel == 0:
        #         ax.set_yticklabels(
        #             [f'$v_{{{i+1}}},\\ \\lambda_{{{i+1}}}={vals[i]:.0f}$' for i in range(6)], fontsize=7)
        #     else:
        #         ax.set_yticklabels(
        #             [f'{vals[i]:.0f}' for i in range(6)], fontsize=7)
        #
        # # colorbar with tick marks at representative |component| values
        # cbar = fig_h.colorbar(im, cax=axes_h[3])
        # cbar.set_label('|component|', fontsize=7)
        # cbar.set_ticks([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0])
        # cbar.ax.tick_params(labelsize=6)
        #
        # plt.tight_layout()

        # --- Eigenvector heatmap at a chosen scan (default: curved_fast degenerate t=688) ---
        THRESH = heat_thresh
        heat_idx = int(np.argmin(np.abs(t_info - heat_t)))

        mat_d = info_mats[heat_idx][:, reorder][reorder, :]
        vals_d, vecs_d = np.linalg.eigh(mat_d)
        data_d = np.abs(vecs_d.T)
        n_below = int(np.sum(vals_d < THRESH))   # weak directions annotated in the title

        fig_d, axes_d = plt.subplots(1, 2, figsize=fig_size(0.5, 1.0),
                                     gridspec_kw={'width_ratios': [4, 0.4]},
                                     layout='none')  # manual cax colorbar incompatible with constrained_layout
        # Two-line title keeps the 'Components of eigenvectors' save-key but fits the
        # 0.5*textwidth canvas; margins leave room for the row labels and colorbar.
        fig_d.subplots_adjust(left=0.30, right=0.86, top=0.80, bottom=0.20, wspace=0.1)
        fig_d.suptitle(f'{ds_name}: Components of Eigenvectors\nat t={t_info[heat_idx]:.1f}s', fontsize=9)
        mark(fig_d, 'Heatmap_eigen')

        ax_d = axes_d[0]
        im_d = ax_d.imshow(data_d, cmap='Greys_r', norm=norm, aspect='auto')

        if vals_d[-1] > 1e-10:
            ratios = np.diff(np.log10(np.clip(vals_d, 1e-10, None)))
            ax_d.axhline(int(np.argmax(ratios)) + 0.5, color='red', lw=1.2)

        ax_d.set_xticks(range(6))
        ax_d.set_xticklabels(col_labels, fontsize=7, rotation=45)
        ax_d.set_yticks(range(6))
        ax_d.set_yticklabels([f'$v_{{{i+1}}},\\ \\lambda_{{{i+1}}}={vals_d[i]:.0f}$' for i in range(6)], fontsize=7)

        cbar_d = fig_d.colorbar(im_d, cax=axes_d[1])
        cbar_d.set_ticks([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0])
        cbar_d.ax.tick_params(labelsize=6)

    else:
        print('Log/pos_log.txt: unexpected column count %d (need >=87).' % pos.shape[1])

except Exception as e:
    print('Could not load Log/pos_log.txt:', e)

### Time log analysis from fast_lio_time_log.csv
try:
    tlog = np.genfromtxt(os.path.join(_dir, 'fast_lio_time_log.csv'),
                         delimiter=',', skip_header=1)
    # columns: 0=timestamp, 1=total, 2=scan_pts, 3=incremental, 4=match_time,
    #          5=del_size, 6=delete, 7=tree_st, 8=tree_end, 9=add_pts,
    #          10=preprocess(lidar cb), 11=solve,
    #          12=imu_proc, 13=fov_seg, 14=downsample, 15=iekf_total
    t_log      = tlog[:, 0] - tlog[0, 0]
    total      = tlog[:, 1]  * 1e3
    match_time = tlog[:, 4]  * 1e3
    incr       = tlog[:, 3]  * 1e3
    delete     = tlog[:, 6]  * 1e3
    n          = len(tlog)
    solve      = tlog[:, 11] * 1e3 if tlog.shape[1] > 11 else np.zeros(n)
    imu_proc   = tlog[:, 12] * 1e3 if tlog.shape[1] > 12 else np.zeros(n)
    fov_seg    = tlog[:, 13] * 1e3 if tlog.shape[1] > 13 else np.zeros(n)
    downsample = tlog[:, 14] * 1e3 if tlog.shape[1] > 14 else np.zeros(n)
    iekf_total = tlog[:, 15] * 1e3 if tlog.shape[1] > 15 else np.zeros(n)
    iekf_other = np.maximum(iekf_total - match_time - solve, 0)
    map_incr   = incr + delete
    other      = np.maximum(total - imu_proc - fov_seg - downsample
                            - iekf_total - map_incr, 0)

    # --- Timing breakdown: single stackplot, 3 largest categories + 'other' ---
    fig, ax = plt.subplots(figsize=fig_size(1.0, 0.45))
    fig.suptitle('FAST-LIO2 Runtime Breakdown')
    mark(fig, 'Timing')

    _bands  = [imu_proc, fov_seg, downsample, match_time, solve, iekf_other, map_incr]
    _labels = ['IEKF Predict', 'fov seg', 'downsample',
               'Point-to-Plane', 'IEKF Update', 'IEKF other',
               'map update (incr+del)']
    _means  = [b.mean() for b in _bands]
    _top3   = sorted(range(len(_bands)), key=lambda k: _means[k], reverse=True)[:3]
    _rest   = [k for k in range(len(_bands)) if k not in _top3]
    # 'other' = the residual (total minus all named bands) + the non-top-3 named bands
    _other_band = other.copy()
    for k in _rest:
        _other_band = _other_band + _bands[k]
    _keep_bands  = [_bands[k] for k in _top3] + [_other_band]
    _keep_labels = [_labels[k] for k in _top3] + ['other']
    print('Timing: kept top 3 = %s; collapsed into "other" = %s + residual'
          % ([_labels[k] for k in _top3], [_labels[k] for k in _rest]))
    _order = np.argsort([b.mean() for b in _keep_bands])
    ax.stackplot(t_log,
                 *[_keep_bands[i] for i in _order],
                 labels=[_keep_labels[i] for i in _order],
                 alpha=0.8)
    ax.set_ylabel('Time [ms]')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid()
    plt.tight_layout()

    # # --- Point cloud and tree sizes ---
    # scan_pts = tlog[:, 2]
    # add_pts  = tlog[:, 9]
    # tree_st  = tlog[:, 7]
    # tree_end = tlog[:, 8]

    # fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    # fig.suptitle('Point Cloud & Tree Sizes (queue=10)')

    # axes[0].plot(t_log, scan_pts, lw=0.7, label='scan points')
    # axes[0].plot(t_log, add_pts,  lw=0.7, label='added to tree')
    # axes[0].set_ylabel('Points')
    # axes[0].legend(fontsize=8)
    # axes[0].grid()

    # axes[1].plot(t_log, tree_st,  lw=0.7, label='tree size (start)')
    # axes[1].plot(t_log, tree_end, lw=0.7, label='tree size (end)')
    # axes[1].set_ylabel('Nodes')
    # axes[1].set_xlabel('Time [s]')
    # axes[1].legend(fontsize=8)
    # axes[1].grid()
    # plt.tight_layout()

except Exception as e:
    print('Could not load fast_lio_time_log.csv:', e)

plt.tight_layout()

if args.out_dir:
    for _num in plt.get_fignums():
        _fig = plt.figure(_num)
        _stem = getattr(_fig, '_save_stem', None)
        if _stem is None:
            continue
        save(_fig, os.path.join(args.out_dir, _stem), tight=_stem not in FIXED_WIDTH_STEMS)
    print('Saved figures to', args.out_dir)
else:
    plt.show()
