import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Log_square_tank_fast')

# --- Time window: edit these or pass as args: plot_zoom.py t_start t_end ---
t_start = float(sys.argv[1]) if len(sys.argv) > 1 else 232.0
t_end   = float(sys.argv[2]) if len(sys.argv) > 2 else 238.0

# --- Load data ---
pre = np.loadtxt(os.path.join(_dir, 'mat_pre.txt'))
out = np.loadtxt(os.path.join(_dir, 'mat_out.txt'))

mask_pre = (pre[:, 0] >= t_start) & (pre[:, 0] <= t_end)
mask_out = (out[:, 0] >= t_start) & (out[:, 0] <= t_end)

tp  = pre[mask_pre, 0]
to  = out[mask_out, 0]

pre_w = pre[mask_pre]
out_w = out[mask_out]

# --- Load pos_log for cost/residual/covariance ---
pos_path = os.path.join(_dir, 'pos_log.txt')
pos = np.loadtxt(pos_path)
mask_pos = (pos[:, 0] >= t_start) & (pos[:, 0] <= t_end)
pos_w = pos[mask_pos]
t_pos = pos_w[:, 0]

# --- Figure 1: Attitude pre vs out ---
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
fig.suptitle('Attitude  (pre=dashed, out=solid)')
labels = ['roll [deg]', 'pitch [deg]', 'yaw [deg]']
for i in range(3):
    axes[i].plot(tp, pre_w[:, i+1], '--', color='C%d' % i, label='pre')
    axes[i].plot(to, out_w[:, i+1], '-',  color='C%d' % i, label='out')
    axes[i].set_ylabel(labels[i])
    axes[i].grid()
    axes[i].legend(fontsize=8)
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()

# --- Figure 1b: Translation pre vs out ---
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
fig.suptitle('Translation  (pre=dashed, out=solid)')
labels_t = ['x [m]', 'y [m]', 'z [m]']
for i in range(3):
    axes[i].plot(tp, pre_w[:, i+4], '--', color='C%d' % i, label='pre')
    axes[i].plot(to, out_w[:, i+4], '-',  color='C%d' % i, label='out')
    axes[i].set_ylabel(labels_t[i])
    axes[i].grid()
    axes[i].legend(fontsize=8)
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()

# --- Figure 2: IEKF correction magnitude ---
n = min(len(tp), len(to))
att_diff = np.linalg.norm(out_w[:n, 1:4] - pre_w[:n, 1:4], axis=1)
pos_diff = np.linalg.norm(out_w[:n, 4:7] - pre_w[:n, 4:7], axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
fig.suptitle('IEKF correction magnitude (out - pre)')
axes[0].plot(tp[:n], att_diff, '-', color='C0')
axes[0].set_ylabel('Attitude correction [deg]')
axes[0].grid()
axes[1].plot(tp[:n], pos_diff, '-', color='C1')
axes[1].set_ylabel('Position correction [m]')
axes[1].grid()
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()

# --- Figure 3: Cost, residual, effective points ---
if pos_w.shape[1] >= 64:
    n_pts    = pos_w[:, 61]
    mean_res = pos_w[:, 62]
    cost     = pos_w[:, 63]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Scan registration quality')
    axes[0].plot(t_pos, cost, '-', color='C3')
    axes[0].set_ylabel('Cost (sum sq residuals)')
    axes[0].grid()
    axes[1].plot(t_pos, mean_res, '-', color='C1')
    axes[1].set_ylabel('Mean residual [m]')
    axes[1].grid()
    axes[2].plot(t_pos, n_pts, '-', color='C2')
    axes[2].set_ylabel('Effective points')
    axes[2].grid()
    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()

# --- Figure 4: Position trajectory ---
fig, ax = plt.subplots(figsize=(8, 8))
fig.suptitle('XY trajectory in window')
ax.plot(out_w[:, 4], out_w[:, 5], '-o', markersize=3, label='out', color='C0')
ax.plot(pre_w[:, 4], pre_w[:, 5], '--', alpha=0.5, label='pre', color='C1')
# Mark spike point
spike_mask = np.abs(tp[:n] - 235.3) < 0.15
if spike_mask.any():
    idx = np.where(spike_mask)[0][0]
    ax.plot(out_w[idx, 4], out_w[idx, 5], 'r*', markersize=15, label='spike (t=235.3)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()
ax.legend()
plt.tight_layout()

# --- Figure 5: Information matrix eigenvalues ---
if pos_w.shape[1] >= 100:
    info_flat = pos_w[:, 64:100]
    info_mats = info_flat.reshape(-1, 6, 6)
    eig_vals = np.zeros((len(t_pos), 6))
    for k in range(len(t_pos)):
        eig_vals[k] = np.linalg.eigvalsh(info_mats[k])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle('Information matrix eigenvalues (ascending)')
    eig_labels = ['eig_1 (min)', 'eig_2', 'eig_3', 'eig_4', 'eig_5', 'eig_6 (max)']
    for i in range(6):
        axes[0].plot(t_pos, eig_vals[:, i], label=eig_labels[i])
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid()
    cond = np.where(eig_vals[:, 5] > 1e-10,
                    eig_vals[:, 0] / eig_vals[:, 5], 0.0)
    axes[1].plot(t_pos, cond, color='C3')
    axes[1].set_ylabel('Condition (eig_min / eig_max)')
    axes[1].grid()
    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()

# --- Figure 6: Per-iteration cost and max|dx| at the spike scan ---
iter_costs_path = os.path.join(_dir, 'debug_iter_costs.txt')
if os.path.exists(iter_costs_path):
    with open(iter_costs_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        # format: time converged cost0 cost1 ... | dx0 dx1 ...
        parts = line.split('|')
        head  = parts[0].split()
        t_scan   = float(head[0])
        by_conv  = int(head[1]) == 1
        costs    = list(map(float, head[2:]))
        max_dxs  = list(map(float, parts[1].split())) if len(parts) > 1 else []

        stop_label = 'converged (dx < 0.001 × 2)' if by_conv else 'max iterations reached'

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig.suptitle('IEKF iterations  t = %.6f' % (t_scan))

        axes[0].plot(range(len(costs)), costs, 'o-', color='C3', label='cost')
        axes[0].set_ylabel('Cost  (Σ residuals²)')
        axes[0].legend(fontsize=8)
        axes[0].grid()

        if max_dxs:
            axes[1].plot(range(len(max_dxs)), max_dxs, 's-', color='C0', label='max|dx|')
            axes[1].axhline(0.001, color='k', linestyle='--', linewidth=0.8, label='threshold 0.001')
            axes[1].set_ylabel('max |dx|  (state update)')
            axes[1].set_yscale('log')
            axes[1].legend(fontsize=8)
            axes[1].grid(which='both')

        axes[-1].set_xlabel('IEKF iteration')
        axes[-1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
else:
    print('debug_iter_costs.txt not found — run the bag with debug_scan_en: true first')

# --- Figure 7: Per-point undistortion corrections (raw vs undistorted, full-res) ---
def _load_pcd_xyz_curv(path):
    """Read a binary PointXYZINormal PCD (32 bytes/point) and return (N,4): x,y,z,curvature."""
    with open(path, 'rb') as f:
        raw = f.read()
    # find DATA binary\n marker
    marker = b'DATA binary\n'
    idx = raw.find(marker)
    if idx < 0:
        return None
    data = raw[idx + len(marker):]
    n_pts = len(data) // 32
    arr = np.frombuffer(data[:n_pts * 32], dtype=np.float32).reshape(n_pts, 8)
    # columns: x y z intensity normal_x normal_y normal_z curvature
    return arr[:, [0, 1, 2, 7]]   # x, y, z, curvature

raw_files   = sorted(glob.glob(os.path.join(_dir, 'debug_scan_raw_*.pcd')))
undist_files = sorted(glob.glob(os.path.join(_dir, 'debug_scan_undist_*.pcd')))

if not raw_files:
    print('Figure 7: no debug_scan_raw_*.pcd found — replay the bag with debug_scan_en: true')
elif not undist_files:
    print('Figure 7: debug_scan_undist_*.pcd not found — rebuild fast_lio and replay the bag')

for raw_path, undist_path in zip(raw_files, undist_files):
    raw_xyzc   = _load_pcd_xyz_curv(raw_path)
    undist_xyzc = _load_pcd_xyz_curv(undist_path)
    if raw_xyzc is None or undist_xyzc is None:
        print(f'Figure 7: could not parse {raw_path} or {undist_path}')
        continue

    # undistorted cloud is sorted by curvature; sort raw to match
    raw_sorted = raw_xyzc[np.argsort(raw_xyzc[:, 3])]
    n = min(len(raw_sorted), len(undist_xyzc))
    raw_s  = raw_sorted[:n]
    und_s  = undist_xyzc[:n]

    t_ms   = und_s[:, 3]          # curvature = ms offset within scan
    corr   = und_s[:, :3] - raw_s[:, :3]   # per-point correction vector
    mag    = np.linalg.norm(corr, axis=1)

    scan_label = os.path.basename(raw_path).replace('debug_scan_raw_', '').replace('.pcd', '')

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'Undistortion corrections  t={scan_label}  (undistorted − raw)')
    axes[0].plot(t_ms, corr[:, 0], ',', markersize=1, color='C0', alpha=0.4)
    axes[0].set_ylabel('Δx [m]')
    axes[0].grid()
    axes[1].plot(t_ms, corr[:, 1], ',', markersize=1, color='C1', alpha=0.4)
    axes[1].set_ylabel('Δy [m]')
    axes[1].grid()
    axes[2].plot(t_ms, corr[:, 2], ',', markersize=1, color='C2', alpha=0.4)
    axes[2].set_ylabel('Δz [m]')
    axes[2].grid()
    axes[3].plot(t_ms, mag, ',', markersize=1, color='C3', alpha=0.4)
    axes[3].set_ylabel('|correction| [m]')
    axes[3].grid()
    axes[-1].set_xlabel('Time within scan [ms]')
    plt.tight_layout()

tick_step = max(0.1, round((t_end - t_start) / 20, 1))
for fig in map(plt.figure, plt.get_fignums()):
    for ax in fig.get_axes():
        if ax.get_xlabel() == 'Time [s]':
            ax.xaxis.set_major_locator(plt.MultipleLocator(tick_step))
    fig.autofmt_xdate(rotation=30)

plt.show()
