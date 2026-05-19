import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spike')

# --- Time window: edit these or pass as args ---
t_start = float(sys.argv[1]) if len(sys.argv) > 1 else 230.0
t_end   = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0

# --- Load data ---
pre = np.loadtxt(os.path.join(_dir, 'mat_pre.txt'))
out = np.loadtxt(os.path.join(_dir, 'mat_out.txt'))

mask_pre = (pre[:, 0] >= t_start) & (pre[:, 0] <= t_end)
mask_out = (out[:, 0] >= t_start) & (out[:, 0] <= t_end)

tp  = pre[mask_pre, 0]
to  = out[mask_out, 0]
pre_w = pre[mask_pre]
out_w = out[mask_out]

# --- Load pos_log ---
pos_path = os.path.join(_dir, 'pos_log.txt')
pos = np.loadtxt(pos_path)
mask_pos = (pos[:, 0] >= t_start) & (pos[:, 0] <= t_end)
pos_w = pos[mask_pos]
t_pos = pos_w[:, 0]

# --- Detect IMU gap from imu.txt (largest gap > 0.5 s in the window) ---
imu_log_path = os.path.join(_dir, 'imu.txt')
gap_start = gap_end = None
if os.path.exists(imu_log_path):
    imu_all = np.loadtxt(imu_log_path)
    imu_tw  = imu_all[(imu_all[:, 0] >= t_start) & (imu_all[:, 0] <= t_end), 0]
    if len(imu_tw) > 1:
        diffs = np.diff(imu_tw)
        big   = np.where(diffs > 0.5)[0]
        if len(big) > 0:
            i = big[np.argmax(diffs[big])]
            gap_start = float(imu_tw[i])
            gap_end   = float(imu_tw[i + 1])

SPIKE_T = 235.3

# --- Helpers ---
def _nan_gaps(t, data, threshold=0.5):
    """Insert NaN rows at time jumps > threshold so matplotlib breaks the line."""
    if len(t) < 2:
        return t.astype(float), np.array(data, dtype=float)
    idx = np.where(np.diff(t) > threshold)[0] + 1
    if len(idx) == 0:
        return t.astype(float), np.array(data, dtype=float)
    t_out = np.insert(t.astype(float), idx, np.nan)
    d = np.array(data, dtype=float)
    if d.ndim == 1:
        d_out = np.insert(d, idx, np.nan)
    else:
        d_out = np.insert(d, idx, np.full((len(idx), d.shape[1]), np.nan), axis=0)
    return t_out, d_out

def _mark_spike(ax, labeled=False):
    lbl = 't=%.1f s' % SPIKE_T if not labeled else '_nolegend_'
    ax.axvline(SPIKE_T, color='red', lw=1.2, zorder=5, label=lbl)
    return True

def _set_half_ticks(axes_list):
    for ax in axes_list:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

def _shade_gap(ax, labeled=False):
    """Shade the IMU gap on an axis. Returns True if label was added."""
    if gap_start is None:
        return labeled
    lbl = 'IMU gap' if not labeled else '_nolegend_'
    ax.axvspan(gap_start, gap_end, color='red', alpha=0.12, label=lbl, zorder=0)
    return True

# Pre-compute NaN-gapped plot arrays
tp_p,    pre_p    = _nan_gaps(tp,    pre_w)
to_p,    out_p    = _nan_gaps(to,    out_w)
t_pos_p, pos_p    = _nan_gaps(t_pos, pos_w)

n = min(len(tp), len(to))

# --- Figure 1: Attitude pre vs out ---
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
fig.suptitle('Attitude  (pre=dashed, out=solid)')
labels = ['roll [deg]', 'pitch [deg]', 'yaw [deg]']
labeled = False
spike_labeled = False
for i in range(3):
    axes[i].plot(tp_p, pre_p[:, i+1], '--', color='C%d' % i, label='pre')
    axes[i].plot(to_p, out_p[:, i+1], '-',  color='C%d' % i, label='out')
    labeled = _shade_gap(axes[i], labeled)
    spike_labeled = _mark_spike(axes[i], spike_labeled)
    axes[i].set_ylabel(labels[i])
    axes[i].grid()
    axes[i].legend(fontsize=8)
axes[-1].set_xlabel('Time [s]')
_set_half_ticks(axes)
plt.tight_layout()

# --- Figure 1b: Translation pre vs out ---
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
fig.suptitle('Translation  (pre=dashed, out=solid)')
labels_t = ['x [m]', 'y [m]', 'z [m]']
labeled = False
spike_labeled = False
for i in range(3):
    axes[i].plot(tp_p, pre_p[:, i+4], '--', color='C%d' % i, label='pre')
    axes[i].plot(to_p, out_p[:, i+4], '-',  color='C%d' % i, label='out')
    labeled = _shade_gap(axes[i], labeled)
    spike_labeled = _mark_spike(axes[i], spike_labeled)
    axes[i].set_ylabel(labels_t[i])
    axes[i].grid()
    axes[i].legend(fontsize=8)
axes[-1].set_xlabel('Time [s]')
_set_half_ticks(axes)
plt.tight_layout()

# --- Figure 2: IEKF correction magnitude ---
att_diff = np.linalg.norm(out_w[:n, 1:4] - pre_w[:n, 1:4], axis=1)
pos_diff = np.linalg.norm(out_w[:n, 4:7] - pre_w[:n, 4:7], axis=1)
tp_n_p, att_p = _nan_gaps(tp[:n], att_diff)
_,       pos_p2 = _nan_gaps(tp[:n], pos_diff)

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
fig.suptitle('IEKF correction magnitude (out - pre)')
labeled = False
spike_labeled = False
axes[0].plot(tp_n_p, att_p,  '-', color='C0')
labeled = _shade_gap(axes[0], labeled)
spike_labeled = _mark_spike(axes[0], spike_labeled)
axes[0].set_ylabel('Attitude correction [deg]')
axes[0].legend(fontsize=8)
axes[0].grid()
axes[1].plot(tp_n_p, pos_p2, '-', color='C1')
labeled = _shade_gap(axes[1], labeled)
_mark_spike(axes[1], spike_labeled)
axes[1].set_ylabel('Position correction [m]')
axes[1].grid()
axes[-1].set_xlabel('Time [s]')
_set_half_ticks(axes)
plt.tight_layout()

# --- Figure 3: Cost, residual, effective points ---
if pos_w.shape[1] >= 64:
    n_pts    = pos_w[:, 61]
    mean_res = pos_w[:, 62]
    cost     = pos_w[:, 63]
    tp3, cost_p    = _nan_gaps(t_pos, cost)
    _,   res_p     = _nan_gaps(t_pos, mean_res)
    _,   npts3_p   = _nan_gaps(t_pos, n_pts)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle('Scan registration quality')
    labeled = False
    spike_labeled = False
    axes[0].plot(tp3, cost_p,  '-', color='C3')
    labeled = _shade_gap(axes[0], labeled)
    spike_labeled = _mark_spike(axes[0], spike_labeled)
    axes[0].set_ylabel('Cost (sum sq residuals)')
    axes[0].legend(fontsize=8)
    axes[0].grid()
    axes[1].plot(tp3, npts3_p, '-', color='C2')
    labeled = _shade_gap(axes[1], labeled)
    _mark_spike(axes[1], spike_labeled)
    axes[1].set_ylabel('Effective points')
    axes[1].grid()
    axes[-1].set_xlabel('Time [s]')
    _set_half_ticks(axes)
    plt.tight_layout()

# --- Figure 4: Position trajectory ---
fig, ax = plt.subplots(figsize=(8, 8))
fig.suptitle('XY trajectory in window')
ax.plot(out_p[:, 4], out_p[:, 5], '-o', markersize=3, label='out', color='C0')
ax.plot(pre_p[:, 4], pre_p[:, 5], '--', alpha=0.5, label='pre', color='C1')

# Mark IMU gap on trajectory
if gap_start is not None:
    before_gap = to < gap_start
    after_gap  = to > gap_end
    if before_gap.any():
        px, py = out_w[before_gap, 4][-1], out_w[before_gap, 5][-1]
        ax.plot(px, py, 'rs', markersize=9, zorder=5, label='gap start (t=%.2f s)' % gap_start)
        ax.annotate('gap start\nt=%.2f s' % gap_start, xy=(px, py),
                    xytext=(8, 8), textcoords='offset points', fontsize=7, color='red')
    if after_gap.any():
        qx, qy = out_w[after_gap, 4][0], out_w[after_gap, 5][0]
        ax.plot(qx, qy, 'r^', markersize=9, zorder=5, label='gap end (t=%.2f s, dur=%.2f s)' % (gap_end, gap_end - gap_start))
        ax.annotate('gap end\nt=%.2f s\ndur=%.2f s' % (gap_end, gap_end - gap_start),
                    xy=(qx, qy), xytext=(8, -28), textcoords='offset points',
                    fontsize=7, color='red')
    if before_gap.any() and after_gap.any():
        ax.plot([px, qx], [py, qy], 'r--', linewidth=1.2, alpha=0.6, zorder=4,
                label='_nolegend_')

spike_t_target = SPIKE_T
idx = int(np.argmin(np.abs(tp[:n] - spike_t_target)))
if np.abs(tp[idx] - spike_t_target) < 1.0:   # only mark if there's a scan within 1 s
    ax.plot(out_w[idx, 4], out_w[idx, 5], 'r*', markersize=15,
            label='spike (t=%.3f)' % tp[idx])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()
ax.legend(fontsize=8)
plt.tight_layout()

# --- Figure: IMU dropout + stale feats_undistort evidence ---
fig, (ax_imu, ax_pts) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
fig.suptitle('IMU dropout and stale scan reuse', fontsize=11)

if os.path.exists(imu_log_path):
    bin_width = 0.5
    bins   = np.arange(t_start, t_end + bin_width, bin_width)
    counts, edges = np.histogram(imu_tw, bins=bins)
    expected = 200.0 * bin_width
    ax_imu.bar(edges[:-1] + bin_width / 2, counts, width=bin_width * 0.85,
               color='C2', alpha=0.75, label='processed IMU msgs')
    ax_imu.axhline(expected, color='k', linestyle='--', linewidth=1.0,
                   label='expected (%d msgs / bin at 200 Hz)' % int(expected))
    _shade_gap(ax_imu, labeled=False)
    ax_imu.set_ylabel('IMU msgs per 0.5 s bin')
    ax_imu.legend(fontsize=8, loc='lower left')
    ax_imu.grid(axis='y', alpha=0.4)
    ax_imu.set_ylim(bottom=0)
else:
    ax_imu.text(0.5, 0.5, 'imu.txt not found', ha='center', va='center',
                transform=ax_imu.transAxes)

npts_w = out[:, -1].astype(int)[mask_out]
to_pts_p, npts_pts_p = _nan_gaps(to, npts_w.astype(float))
ax_pts.plot(to_pts_p, npts_pts_p, 'o-', markersize=3, linewidth=1, color='C0',
            label='feats_undistort point count')
_shade_gap(ax_pts, labeled=True)

# Show frozen count only if old (unfixed) data is present
frozen_mask = (to >= (gap_start or 0) - 0.01) & (to <= (gap_end or 0) + 0.01)
if frozen_mask.any():
    vals_in_gap = npts_w[frozen_mask]
    if np.all(vals_in_gap == vals_in_gap[0]):
        ax_pts.axhline(int(vals_in_gap[0]), color='red', linestyle='--',
                       linewidth=0.9, alpha=0.8,
                       label='frozen count = %d (stale scan reused)' % int(vals_in_gap[0]))

ax_pts.set_xlabel('Time [s]')
ax_pts.set_ylabel('Points in scan')
ax_pts.legend(fontsize=8, loc='lower left')
ax_pts.grid(alpha=0.4)
_mark_spike(ax_imu, False)
_mark_spike(ax_pts, True)
_set_half_ticks([ax_imu, ax_pts])
plt.tight_layout()

# --- Figure 5: Information matrix eigenvalues ---
if pos_w.shape[1] >= 87:
    info_mats = pos_w[:, 51:87].reshape(-1, 6, 6)
    eig_vals  = np.array([np.linalg.eigvalsh(m) for m in info_mats])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle('Information matrix eigenvalues (ascending)')
    eig_labels = ['eig_1 (min)', 'eig_2', 'eig_3', 'eig_4', 'eig_5', 'eig_6 (max)']
    labeled = False
    spike_labeled = False
    for i in range(6):
        tp5, ev_p = _nan_gaps(t_pos, eig_vals[:, i])
        axes[0].plot(tp5, ev_p, label=eig_labels[i])
    labeled = _shade_gap(axes[0], labeled)
    spike_labeled = _mark_spike(axes[0], spike_labeled)
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid()

    cond = np.where(eig_vals[:, 5] > 1e-10,
                    eig_vals[:, 0] / eig_vals[:, 5], 0.0)
    tp5c, cond_p = _nan_gaps(t_pos, cond)
    axes[1].plot(tp5c, cond_p, color='C3')
    _shade_gap(axes[1], labeled)
    _mark_spike(axes[1], spike_labeled)
    axes[1].set_ylabel('Condition (eig_min / eig_max)')
    axes[1].grid()
    axes[-1].set_xlabel('Time [s]')
    _set_half_ticks(axes)
    plt.tight_layout()

# --- Figure 6a: Eigenvector heatmaps around the spike ---
if pos_w.shape[1] >= 87:
    info_mats = pos_w[:, 51:87].reshape(-1, 6, 6)
    spike_t   = SPIKE_T
    n_side    = 3
    spike_idx = int(np.argmin(np.abs(t_pos - spike_t)))
    sel = list(range(max(0, spike_idx - n_side),
                     min(len(t_pos), spike_idx + n_side + 1)))

    reorder    = [3, 4, 5, 0, 1, 2]
    col_labels = ['Rx', 'Ry', 'Rz', 'X', 'Y', 'Z']
    n_panels   = len(sel)
    fig, axes  = plt.subplots(1, n_panels, figsize=(2.2 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle('Eigenvector heatmaps  (rows=v1..v6 ascending eigenvalue, cols=state DOF)\n'
                 'recovery scan at t=%.3fs highlighted in red' % spike_t, fontsize=9)

    for panel_i, idx in enumerate(sel):
        mat  = info_mats[idx][:, reorder][reorder, :]
        vals, vecs = np.linalg.eigh(mat)
        data = np.abs(vecs.T)
        ax   = axes[panel_i]
        ax.imshow(data, cmap='Greys_r', vmin=0, vmax=1, aspect='auto')
        is_spike = (idx == spike_idx)
        ax.set_title('t=%.2f' % t_pos[idx], fontsize=7,
                     color='red' if is_spike else 'black',
                     fontweight='bold' if is_spike else 'normal')
        ax.set_xticks(range(6))
        ax.set_xticklabels(col_labels, fontsize=6, rotation=45)
        ax.set_yticks(range(6))
        if panel_i == 0:
            ax.set_yticklabels(['v%d  %.0f' % (i+1, vals[i]) for i in range(6)], fontsize=6)
        else:
            ax.set_yticklabels(['%.0f' % vals[i] for i in range(6)], fontsize=6)
        if vals[-1] > 1e-10:
            ratios = np.diff(np.log10(np.clip(vals, 1e-10, None)))
            gap_row = int(np.argmax(ratios))
            ax.axhline(gap_row + 0.5, color='red', linewidth=1.2)
        for spine in ax.spines.values():
            spine.set_edgecolor('red' if is_spike else 'grey')
            spine.set_linewidth(2 if is_spike else 0.5)
    plt.tight_layout()

# --- Figure 6: Per-iteration cost and max|dx| at the spike scan ---
iter_costs_path = os.path.join(_dir, 'debug_iter_costs.txt')
if os.path.exists(iter_costs_path):
    with open(iter_costs_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        parts   = line.split('|')
        head    = parts[0].split()
        t_scan  = float(head[0])
        by_conv = int(head[1]) == 1
        costs   = list(map(float, head[2:]))
        max_dxs = list(map(float, parts[1].split())) if len(parts) > 1 else []

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig.suptitle('IEKF iterations  t = %.6f' % t_scan)
        axes[0].plot(range(len(costs)), costs, 'o-', color='C3', label='cost')
        axes[0].set_ylabel('Cost  (sum residuals^2)')
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

# --- Figure: IMU propagation steps through post-gap scan (scan_dbg.txt) ---
scan_dbg_path = os.path.join(_dir, 'scan_dbg.txt')
if os.path.exists(scan_dbg_path):
    sequences, cur_scan, cur_props = [], None, []
    with open(scan_dbg_path) as f:
        for line in f:
            parts = dict(p.split('=') for p in line.split() if '=' in p)
            if line.startswith('SCAN'):
                if cur_scan is not None and cur_props:
                    sequences.append({'scan': cur_scan, 'props': np.array(cur_props)})
                cur_scan = {'t': float(parts['t']), 'imu_n': int(parts['imu_n']),
                            'gap_s': float(parts['gap_s'])}
                cur_props = []
            elif line.startswith('PROP') and cur_scan is not None:
                cur_props.append([
                    int(parts['step']),
                    float(parts['t']),    float(parts['dt']),
                    float(parts['px']),   float(parts['py']),   float(parts['pz']),
                    float(parts['roll']), float(parts['pitch']),float(parts['yaw']),
                    float(parts['vx']),   float(parts['vy']),   float(parts['vz']),
                ])
    if cur_scan is not None and cur_props:
        sequences.append({'scan': cur_scan, 'props': np.array(cur_props)})

    sequences = [s for s in sequences if t_start <= s['scan']['t'] <= t_end]

    if sequences:
        for seq in sequences:
            sc    = seq['scan']
            props = seq['props']
            # cols: step, t, dt, px, py, pz, roll, pitch, yaw, vx, vy, vz
            step_ids = props[:, 0].astype(int)   # -1, 0, 1, ..., N
            idx      = np.arange(len(step_ids))  # 0, 1, 2, ... for uniform x-axis
            t_prop   = props[:, 1]
            dt_prop  = props[:, 2]
            px, py, pz       = props[:, 3], props[:, 4], props[:, 5]
            roll, pitch, yaw = props[:, 6], props[:, 7], props[:, 8]
            vx, vy, vz       = props[:, 9], props[:,10], props[:,11]

            # Index of the large-dt gap step (first step with dt > 0.1s)
            gap_idx = int(np.argmax(dt_prop > 0.1)) if np.any(dt_prop > 0.1) else None

            # Total change from initial state (step -1)
            d_pos  = np.sqrt((px-px[0])**2 + (py-py[0])**2 + (pz-pz[0])**2)
            d_vel  = np.sqrt((vx-vx[0])**2 + (vy-vy[0])**2 + (vz-vz[0])**2)
            # Euler angle differences with wrap-around correction to [-180, 180]
            d_roll  = ((roll  - roll[0]  + 180) % 360) - 180
            d_pitch = ((pitch - pitch[0] + 180) % 360) - 180
            d_yaw   = ((yaw   - yaw[0]   + 180) % 360) - 180
            d_att   = np.sqrt(d_roll**2 + d_pitch**2 + d_yaw**2)

            # x-tick labels: show actual time at each step
            tick_labels = ['%.3f\n(dt=%.3fs)' % (t_prop[i], dt_prop[i])
                           if gap_idx is not None and i == gap_idx
                           else '%.3f' % t_prop[i]
                           for i in range(len(idx))]
            # Only label a subset to avoid clutter
            sparse = max(1, len(idx) // 8)
            shown  = set(range(0, len(idx), sparse)) | {0, gap_idx} if gap_idx is not None \
                     else set(range(0, len(idx), sparse)) | {0}
            shown.discard(None)

            def _set_step_xticks(ax):
                ax.set_xticks(idx[list(shown)])
                ax.set_xticklabels([tick_labels[i] for i in sorted(shown)],
                                   fontsize=7, rotation=30, ha='right')
                if gap_idx is not None:
                    ax.axvline(gap_idx, color='red', ls='--', lw=1, alpha=0.6,
                               label='gap step (dt=%.3fs)' % dt_prop[gap_idx])
                ax.set_xlabel('Step index  (label = rel. time [s])')

            bar_colors = ['red' if dt_prop[i] > 0.1 else 'steelblue' for i in range(len(dt_prop))]

            fig, axes = plt.subplots(3, 2, figsize=(15, 11))
            fig.suptitle('IMU propagation after %.2f s gap  (scan t=%.3f s,  %d IMU steps)'
                         % (sc['gap_s'], sc['t'], sc['imu_n']), fontsize=11)

            # --- Total displacement from initial state ---
            axes[0,0].plot(idx, d_pos, 'o-', ms=4, color='C0')
            axes[0,0].set_title('Total displacement from initial state [m]')
            axes[0,0].set_ylabel('|Δpos| [m]')
            axes[0,0].grid()
            _set_step_xticks(axes[0,0])

            # --- XY path zoomed to propagation region ---
            pad = 0.3
            ax = axes[0,1]
            ax.plot(px, py, 'o-', color='C3', lw=1, ms=4)
            ax.plot(px[0],  py[0],  'gs', markersize=9, zorder=5, label='start (pre-gap state)')
            if gap_idx is not None:
                ax.plot(px[gap_idx], py[gap_idx], 'r^', markersize=9, zorder=5,
                        label='after gap step (dt=%.2fs)' % dt_prop[gap_idx])
            ax.plot(px[-1], py[-1], 'kD', markersize=7, zorder=5, label='final (scan pre)')
            ax.set_xlim(px.min()-pad, px.max()+pad)
            ax.set_ylim(py.min()-pad, py.max()+pad)
            ax.set_title('XY propagation path (zoomed)')
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
            ax.set_aspect('equal'); ax.legend(fontsize=8); ax.grid()

            # --- Total orientation change from initial state ---
            axes[1,0].plot(idx, d_att, 'o-', ms=4, color='C1')
            axes[1,0].set_title('Total orientation change from initial state [deg]')
            axes[1,0].set_ylabel('|Δatt| [deg]')
            axes[1,0].grid()
            _set_step_xticks(axes[1,0])

            # --- Total velocity change vs step ---
            axes[1,1].plot(idx, d_vel, 'o-', ms=4, color='C2')
            axes[1,1].set_title('Total velocity change from initial state [m/s]')
            axes[1,1].set_ylabel('|Δvel| [m/s]')
            axes[1,1].grid()
            _set_step_xticks(axes[1,1])

            # --- dt bar ---
            axes[2,0].bar(idx, dt_prop, color=bar_colors)
            axes[2,0].set_title('dt per propagation step [s]  (red = gap step)')
            axes[2,0].set_xlabel('Step index'); axes[2,0].set_ylabel('dt [s]')
            axes[2,0].grid(axis='y')

            # --- |Δpos| bar ---
            dp = np.sqrt(np.diff(px)**2 + np.diff(py)**2 + np.diff(pz)**2)
            dp_colors = ['red' if (gap_idx is not None and i == gap_idx - 1)
                         else 'steelblue' for i in range(len(dp))]
            axes[2,1].bar(range(len(dp)), dp, color=dp_colors)
            axes[2,1].set_title('|Δpos| per step [m]  (red = gap step)')
            axes[2,1].set_xlabel('Step index'); axes[2,1].set_ylabel('Distance [m]')
            axes[2,1].grid(axis='y')

            plt.tight_layout()
    else:
        print('scan_dbg.txt: no post-gap scans in window [%.1f, %.1f]' % (t_start, t_end))
else:
    print('scan_dbg.txt not found — run the bag with debug_scan_en: true first')

plt.show()
