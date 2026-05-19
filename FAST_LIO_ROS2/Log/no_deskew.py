import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_base  = os.path.dirname(os.path.abspath(__file__))
dk_dir = os.path.join(_base, 'Log_curved_tank_fast')
nd_dir = os.path.join(_base, 'Log_no_motion_deskew', 'curved_tank_fast')

def norm_t(t):
    return t - t[0]

xyz = ['x', 'y', 'z']

# ── ikfom state logs ───────────────────────────────────────────────────────────
dk_pre = np.loadtxt(os.path.join(dk_dir, 'mat_pre.txt'))
dk_out = np.loadtxt(os.path.join(dk_dir, 'mat_out.txt'))
nd_pre = np.loadtxt(os.path.join(nd_dir, 'mat_pre.txt'))
nd_out = np.loadtxt(os.path.join(nd_dir, 'mat_out.txt'))
dk_t = norm_t(dk_pre[:, 0])
nd_t = norm_t(nd_pre[:, 0])

# ── State comparison (IEKF output) ─────────────────────────────────────────────
state_cfg = [
    (0, 'Attitude',              'deg'),
    (1, 'Translation',           'm'),
    (4, 'Velocity',              'm/s'),
    (5, 'Bias of Gyroscope',     'rad/s'),
    (6, 'Bias of Accelerometer', 'm/s²'),
    (7, 'Estimated Gravity',     'm/s²'),
]

for j, title, unit in state_cfg:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f'{title}  |  C0-C2 = deskew,  C3-C5 = no-deskew')
    for i in range(3):
        col = 1 + j * 3 + i
        ax.plot(dk_t, dk_out[:, col], color=f'C{i}',   label=f'{xyz[i]} (deskew)')
        ax.plot(nd_t, nd_out[:, col], color=f'C{i+3}', label=f'{xyz[i]} (no-deskew)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'[{unit}]')
    ax.grid()
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

# ── XY trajectory ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
fig.suptitle('Estimated Trajectory (XY)')
ax.plot(dk_out[:, 4], dk_out[:, 5], color='C0', lw=1.2, label='deskew')
ax.plot(nd_out[:, 4], nd_out[:, 5], color='C1', lw=1.2, label='no-deskew')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()
ax.legend()

# ── 3D trajectory ──────────────────────────────────────────────────────────────
fig3d = plt.figure()
ax3d  = fig3d.add_subplot(111, projection='3d')
fig3d.suptitle('Estimated Trajectory (3D)')
ax3d.plot(dk_out[:, 4], dk_out[:, 5], dk_out[:, 6], color='C0', lw=0.8, label='deskew')
ax3d.plot(nd_out[:, 4], nd_out[:, 5], nd_out[:, 6], color='C1', lw=0.8, label='no-deskew')
ax3d.set_xlabel('x [m]'); ax3d.set_ylabel('y [m]'); ax3d.set_zlabel('z [m]')
ax3d.legend(); ax3d.grid()
all_x = np.concatenate([dk_out[:, 4], nd_out[:, 4]])
all_y = np.concatenate([dk_out[:, 5], nd_out[:, 5]])
all_z = np.concatenate([dk_out[:, 6], nd_out[:, 6]])
pad = 0.5
ax3d.set_xlim(all_x.min()-pad, all_x.max()+pad)
ax3d.set_ylim(all_y.min()-pad, all_y.max()+pad)
ax3d.set_zlim(all_z.min()-pad, all_z.max()+pad)

# ── Absolute Trajectory Error (deskew vs no-deskew) ───────────────────────────
# Interpolate nd to dk timestamps, then align via SVD (optimal R + t)
dk_xyz = dk_out[:, 4:7]
nd_xyz = nd_out[:, 4:7]

common_min = max(dk_t[0], nd_t[0])
common_max = min(dk_t[-1], nd_t[-1])
mask_ate   = (dk_t >= common_min) & (dk_t <= common_max)
t_ref      = dk_t[mask_ate]
p_ref      = dk_xyz[mask_ate]
p_nd_at    = np.column_stack([np.interp(t_ref, nd_t, nd_xyz[:, j]) for j in range(3)])

mu_ref  = p_ref.mean(axis=0)
mu_nd   = p_nd_at.mean(axis=0)
H       = (p_nd_at - mu_nd).T @ (p_ref - mu_ref)
U, _, Vt = np.linalg.svd(H)
R_ate   = Vt.T @ U.T
if np.linalg.det(R_ate) < 0:
    Vt[-1] *= -1
    R_ate = Vt.T @ U.T
t_ate        = mu_ref - R_ate @ mu_nd
p_nd_aligned = (R_ate @ p_nd_at.T).T + t_ate

err_vec   = p_ref - p_nd_aligned
err_norm  = np.linalg.norm(err_vec, axis=1)
ate_mean  = float(err_norm.mean())
ate_final = float(err_norm[-1])

fig_ate, axes_ate = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig_ate.suptitle(f'Absolute Trajectory Error — deskew vs no-deskew'
                 f'  (mean = {ate_mean:.4f} m,  final = {ate_final:.4f} m)')

axes_ate[0].plot(t_ref, err_norm, color='C3', lw=0.8, label='|error|')
axes_ate[0].axhline(ate_mean,  color='r',  ls='--', lw=1, label='mean %.4f m'  % ate_mean)
axes_ate[0].axhline(ate_final, color='k',  ls=':',  lw=1, label='final %.4f m' % ate_final)
axes_ate[0].set_ylabel('Position error [m]')
axes_ate[0].legend(fontsize=8)
axes_ate[0].grid()

for j, (lbl, col) in enumerate(zip(['x', 'y', 'z'], ['C0', 'C1', 'C2'])):
    axes_ate[1].plot(t_ref, np.abs(err_vec[:, j]), color=col, lw=0.8, label=f'{lbl} error')
axes_ate[1].set_ylabel('|Component error| [m]')
axes_ate[1].set_xlabel('Time [s]')
axes_ate[1].legend(fontsize=8)
axes_ate[1].grid()

plt.tight_layout()

# ── pos_log: covariance & information matrix ───────────────────────────────────
def load_pos(d):
    p = np.loadtxt(os.path.join(d, 'pos_log.txt'))
    return p.reshape(1, -1) if p.ndim == 1 else p

try:
    dk_pos = load_pos(dk_dir)
    nd_pos = load_pos(nd_dir)
    dk_tp  = norm_t(dk_pos[:, 0])
    nd_tp  = norm_t(nd_pos[:, 0])

    if dk_pos.shape[1] >= 61 and nd_pos.shape[1] >= 61:
        cov_labels = ['cov_x', 'cov_y', 'cov_z', 'cov_roll', 'cov_pitch', 'cov_yaw']
        fig_c, ax_c = plt.subplots(2, 3, figsize=(12, 6))
        fig_c.suptitle('Pose covariance diagonals  |  C0 = deskew,  C1 = no-deskew')
        ax_c = ax_c.ravel()
        for run_pos, run_t, color, lbl in [
            (dk_pos, dk_tp, 'C0', 'deskew'),
            (nd_pos, nd_tp, 'C1', 'no-deskew'),
        ]:
            cov  = run_pos[:, 25:61].reshape(-1, 6, 6)
            diag = np.array([cov[:, i, i] for i in range(6)])
            for i in range(6):
                ax_c[i].plot(run_t, diag[i], color=color, label=lbl)
                ax_c[i].axhline(diag[i].mean(), color=color, ls='--', lw=1, label=f'{lbl} mean')
                ax_c[i].set_title(cov_labels[i])
                ax_c[i].grid(True)
        ax_c[2].legend(fontsize=7)
        plt.tight_layout()

    if dk_pos.shape[1] >= 100 and nd_pos.shape[1] >= 100:
        def compute_info(pos):
            info_mats = pos[:, 64:100].reshape(-1, 6, 6)
            n = len(info_mats)
            eig_vals = np.zeros((n, 6))
            eig_vecs = np.zeros((n, 6, 6))
            for k in range(n):
                vals, vecs = np.linalg.eigh(info_mats[k])
                eig_vals[k] = vals
                eig_vecs[k] = vecs
            cond = np.where(eig_vals[:, 5] > 1e-10,
                            eig_vals[:, 0] / eig_vals[:, 5], 0.0)
            return eig_vals, eig_vecs, cond

        dk_eig_vals, dk_eig_vecs, dk_cond = compute_info(dk_pos)
        nd_eig_vals, nd_eig_vecs, nd_cond = compute_info(nd_pos)

        fig_i, axs_i = plt.subplots(1, 3, figsize=(16, 4))
        fig_i.suptitle('IEKF Information Matrix  |  C0 = deskew,  C1 = no-deskew')

        dk_npts = dk_pos[:, 61]; nd_npts = nd_pos[:, 61]
        axs_i[0].plot(dk_tp, dk_npts, color='C0', label='deskew')
        axs_i[0].plot(nd_tp, nd_npts, color='C1', label='no-deskew')
        axs_i[0].axhline(dk_npts.mean(), color='C0', ls='--', lw=1, label='dk mean %.0f' % dk_npts.mean())
        axs_i[0].axhline(nd_npts.mean(), color='C1', ls='--', lw=1, label='nd mean %.0f' % nd_npts.mean())
        axs_i[0].set_title('Effective Points')
        axs_i[0].grid(); axs_i[0].legend(fontsize=7)

        dk_cost = dk_pos[:, 63]; nd_cost = nd_pos[:, 63]
        axs_i[1].plot(dk_tp, dk_cost, color='C0', label='deskew')
        axs_i[1].plot(nd_tp, nd_cost, color='C1', label='no-deskew')
        axs_i[1].axhline(dk_cost.mean(), color='C0', ls='--', lw=1, label='dk mean %.3g' % dk_cost.mean())
        axs_i[1].axhline(nd_cost.mean(), color='C1', ls='--', lw=1, label='nd mean %.3g' % nd_cost.mean())
        axs_i[1].set_title('Cost (sum sq res)')
        axs_i[1].grid(); axs_i[1].legend(fontsize=7)

        axs_i[2].plot(dk_tp, dk_cond, color='C0', label='deskew')
        axs_i[2].plot(nd_tp, nd_cond, color='C1', label='no-deskew')
        axs_i[2].axhline(dk_cond.mean(), color='C0', ls='--', lw=1, label='dk mean %.3g' % dk_cond.mean())
        axs_i[2].axhline(nd_cond.mean(), color='C1', ls='--', lw=1, label='nd mean %.3g' % nd_cond.mean())
        axs_i[2].set_title('Condition Number (eig_min / eig_max)')
        axs_i[2].grid(); axs_i[2].legend(fontsize=7)

        for ax in axs_i:
            ax.set_xlabel('Time [s]')
        plt.tight_layout()

        # Weakest eigenvector components — side by side
        pose_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z']
        fig_w, axes_w = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        fig_w.suptitle('Weakest Eigenvector Components (eig_1 direction)')
        for ax, eig_vecs, run_t, label in [
            (axes_w[0], dk_eig_vecs, dk_tp, 'deskew'),
            (axes_w[1], nd_eig_vecs, nd_tp, 'no-deskew'),
        ]:
            weakest = eig_vecs[:, :, 0]
            for i in range(6):
                ax.plot(run_t, np.abs(weakest[:, i]), label=pose_labels[i])
            ax.set_title(label)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('|component|')
            ax.grid(); ax.legend(fontsize=7)
        plt.tight_layout()

except Exception as e:
    print('Could not load pos_log.txt:', e)

# # ── Timing comparison ─────────────────────────────────────────────────────────
# def load_tlog(d):
#     tlog = np.genfromtxt(os.path.join(d, 'fast_lio_time_log.csv'),
#                          delimiter=',', skip_header=1)
#     n = len(tlog)
#     def ms(idx):
#         return tlog[:, idx] * 1e3 if tlog.shape[1] > idx else np.zeros(n)
#     return dict(
#         t        = norm_t(tlog[:, 0]),
#         total    = ms(1),
#         scan_pts = tlog[:, 2],
#         incr     = ms(3),
#         match    = ms(4),
#         delete   = ms(6),
#         tree_st  = tlog[:, 7],
#         tree_end = tlog[:, 8],
#         add_pts  = tlog[:, 9],
#         solve    = ms(11),
#         imu_proc = ms(12),
#         fov_seg  = ms(13),
#         downsamp = ms(14),
#         iekf_tot = ms(15),
#     )

# try:
#     dk_tl = load_tlog(dk_dir)
#     nd_tl = load_tlog(nd_dir)

#     # Total time overlay
#     fig, ax = plt.subplots(figsize=(14, 4))
#     fig.suptitle('FAST-LIO Total Processing Time')
#     ax.plot(dk_tl['t'], dk_tl['total'], lw=0.7, color='C0', label='deskew')
#     ax.plot(nd_tl['t'], nd_tl['total'], lw=0.7, color='C1', label='no-deskew')
#     ax.axhline(dk_tl['total'].mean(), color='C0', ls=':', lw=1,
#                label=f'dk mean {dk_tl["total"].mean():.1f} ms')
#     ax.axhline(nd_tl['total'].mean(), color='C1', ls=':', lw=1,
#                label=f'nd mean {nd_tl["total"].mean():.1f} ms')
#     ax.axhline(100, color='k', ls='-', lw=0.8, label='scan period 100 ms')
#     ax.set_ylabel('Time [ms]'); ax.set_xlabel('Time [s]')
#     ax.legend(fontsize=8); ax.grid()
#     plt.tight_layout()

#     # Timing breakdown — stacked area side by side
#     def stacked_timing(ax, tl, label):
#         t         = tl['t']
#         iekf_other = np.maximum(tl['iekf_tot'] - tl['match'] - tl['solve'], 0)
#         map_incr   = tl['incr'] + tl['delete']
#         other      = np.maximum(tl['total'] - tl['imu_proc'] - tl['fov_seg']
#                                 - tl['downsamp'] - tl['iekf_tot'] - map_incr, 0)
#         threshold = 0.01 * tl['total'].mean()
#         bands_all  = [tl['imu_proc'], tl['fov_seg'], tl['downsamp'],
#                       tl['match'], tl['solve'], iekf_other, map_incr, other]
#         labels_all = ['imu proc', 'fov seg', 'downsample',
#                       'iekf match', 'iekf solve', 'iekf other',
#                       'map update', 'other']
#         merged = other.copy()
#         keep_b, keep_l = [], []
#         for b, l in zip(bands_all[:-1], labels_all[:-1]):
#             if b.mean() < threshold:
#                 merged += b
#             else:
#                 keep_b.append(b)
#                 keep_l.append(l)
#         keep_b.append(merged); keep_l.append('other')
#         order = np.argsort([b.mean() for b in keep_b])
#         ax.stackplot(t, *[keep_b[i] for i in order],
#                      labels=[keep_l[i] for i in order], alpha=0.8)
#         ax.set_title(label)
#         ax.set_ylabel('Time [ms]'); ax.set_xlabel('Time [s]')
#         ax.legend(fontsize=7, loc='upper left'); ax.grid()

#     fig_s, (ax_dk, ax_nd) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
#     fig_s.suptitle('FAST-LIO Timing Breakdown')
#     stacked_timing(ax_dk, dk_tl, 'deskew')
#     stacked_timing(ax_nd, nd_tl, 'no-deskew')
#     plt.tight_layout()

#     # Box plot comparison of timing components
#     comp_keys  = ['total', 'iekf_tot', 'match', 'solve', 'imu_proc', 'fov_seg', 'downsamp', 'incr', 'delete']
#     comp_names = ['total', 'iekf',     'match', 'solve', 'imu_proc', 'fov_seg', 'downsamp', 'incr', 'delete']
#     n_comp = len(comp_keys)
#     fig_box, ax_box = plt.subplots(figsize=(12, 5))
#     fig_box.suptitle('Timing Components — box plots (no outliers)')
#     pos_dk = np.arange(n_comp) * 3
#     pos_nd = np.arange(n_comp) * 3 + 1
#     ax_box.boxplot([dk_tl[k] for k in comp_keys], positions=pos_dk, showfliers=False,
#                    widths=0.8, patch_artist=True,
#                    boxprops=dict(facecolor='C0', alpha=0.5),
#                    medianprops=dict(color='C0', lw=2))
#     ax_box.boxplot([nd_tl[k] for k in comp_keys], positions=pos_nd, showfliers=False,
#                    widths=0.8, patch_artist=True,
#                    boxprops=dict(facecolor='C1', alpha=0.5),
#                    medianprops=dict(color='C1', lw=2))
#     ax_box.set_xticks(np.arange(n_comp) * 3 + 0.5)
#     ax_box.set_xticklabels(comp_names, fontsize=8)
#     ax_box.set_ylabel('Time [ms]')
#     ax_box.legend(handles=[Patch(facecolor='C0', alpha=0.5, label='deskew'),
#                             Patch(facecolor='C1', alpha=0.5, label='no-deskew')], fontsize=8)
#     ax_box.grid(axis='y')
#     plt.tight_layout()

#     # Point cloud & tree sizes
#     fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
#     fig.suptitle('Point Cloud & Tree Sizes')
#     axes[0].plot(dk_tl['t'], dk_tl['scan_pts'], lw=0.7, color='C0', label='scan pts (deskew)')
#     axes[0].plot(nd_tl['t'], nd_tl['scan_pts'], lw=0.7, color='C1', label='scan pts (no-deskew)')
#     axes[0].plot(dk_tl['t'], dk_tl['add_pts'],  lw=0.7, color='C2', label='added pts (deskew)')
#     axes[0].plot(nd_tl['t'], nd_tl['add_pts'],  lw=0.7, color='C3', label='added pts (no-deskew)')
#     axes[0].set_ylabel('Points'); axes[0].legend(fontsize=7); axes[0].grid()
#     axes[1].plot(dk_tl['t'], dk_tl['tree_st'],  lw=0.7, color='C0', label='tree start (deskew)')
#     axes[1].plot(nd_tl['t'], nd_tl['tree_st'],  lw=0.7, color='C1', label='tree start (no-deskew)')
#     axes[1].plot(dk_tl['t'], dk_tl['tree_end'], lw=0.7, color='C2', label='tree end (deskew)')
#     axes[1].plot(nd_tl['t'], nd_tl['tree_end'], lw=0.7, color='C3', label='tree end (no-deskew)')
#     axes[1].set_ylabel('Nodes'); axes[1].set_xlabel('Time [s]')
#     axes[1].legend(fontsize=7); axes[1].grid()
#     plt.tight_layout()

# except Exception as e:
#     print('Could not load fast_lio_time_log.csv:', e)

# # ── No-deskew eigenvector heatmaps at deskew peak timestamps ──────────────────
# try:
#     from matplotlib.colors import PowerNorm
#     nd_pos_h   = load_pos(nd_dir)
#     if nd_pos_h.shape[1] >= 100:
#         t_nd_h       = norm_t(nd_pos_h[:, 0])
#         info_mats_nd = nd_pos_h[:, 64:100].reshape(-1, 6, 6)

#         target_times  = [50.3, 161.9, 428.7]
#         target_labels = ['X dominant (dk t=50.3s)', 'Y dominant (dk t=161.9s)', 'Z dominant (dk t=428.7s)']
#         reorder    = [3, 4, 5, 0, 1, 2]
#         col_labels = ['Rx', 'Ry', 'Rz', 'X', 'Y', 'Z']
#         norm_pw    = PowerNorm(gamma=0.35, vmin=0, vmax=1)

#         fig_h, axes_h = plt.subplots(1, 4, figsize=(12, 5),
#                                      gridspec_kw={'width_ratios': [4, 4, 4, 0.4]})
#         fig_h.suptitle('No-deskew eigenvector heatmaps at deskew peak X/Y/Z dominance times\n'
#                         '(rows = v1..v6 ascending eigenvalue, cols = Rx Ry Rz X Y Z)', fontsize=9)

#         for panel, (t_target, t_label) in enumerate(zip(target_times, target_labels)):
#             idx = int(np.argmin(np.abs(t_nd_h - t_target)))
#             mat = info_mats_nd[idx][:, reorder][reorder, :]
#             vals, vecs = np.linalg.eigh(mat)
#             data = np.abs(vecs.T)

#             ax = axes_h[panel]
#             im = ax.imshow(data, cmap='Greys_r', norm=norm_pw, aspect='auto')

#             if vals[-1] > 1e-10:
#                 ratios = np.diff(np.log10(np.clip(vals, 1e-10, None)))
#                 ax.axhline(int(np.argmax(ratios)) + 0.5, color='red', lw=1.2)

#             ax.set_title(f'{t_label}\nnd t={t_nd_h[idx]:.2f}s', fontsize=8)
#             ax.set_xticks(range(6))
#             ax.set_xticklabels(col_labels, fontsize=7, rotation=45)
#             ax.set_yticks(range(6))
#             if panel == 0:
#                 ax.set_yticklabels([f'v{i+1}  {vals[i]:.0f}' for i in range(6)], fontsize=7)
#             else:
#                 ax.set_yticklabels([f'{vals[i]:.0f}' for i in range(6)], fontsize=7)

#         cbar = fig_h.colorbar(im, cax=axes_h[3])
#         cbar.set_label('|component|', fontsize=7)
#         cbar.set_ticks([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0])
#         cbar.ax.tick_params(labelsize=6)
#         plt.tight_layout()

# except Exception as e:
#     print('Could not generate no-deskew heatmap:', e)

plt.show()
