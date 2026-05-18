# import matplotlib
# matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt

_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Log_square_tank_slow') # adjust to your log directory


#######for ikfom
lab_pre = ['', 'pre-x', 'pre-y', 'pre-z']
lab_out = ['', 'out-x', 'out-y', 'out-z']
a_pre=np.loadtxt(os.path.join(_dir, 'mat_pre.txt'))
a_out=np.loadtxt(os.path.join(_dir, 'mat_out.txt'))
time=a_pre[:,0]

# --- Attitude (j=0) ---
fig, ax = plt.subplots()
fig.suptitle('Attitude')
for i in range(1, 4):
    ax.plot(time, a_pre[:, i+0*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+0*3], '-', label=lab_out[i])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Attitude [deg]')
ax.grid()
ax.legend()

# --- Translation (j=1) ---
fig, ax = plt.subplots()
fig.suptitle('Translation')
for i in range(1, 4):
    ax.plot(time, a_pre[:, i+1*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+1*3], '-', label=lab_out[i])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Translation [m]')
ax.grid()
ax.legend()

# --- XY trajectory ---
fig, ax = plt.subplots()
fig.suptitle('Estimated Trajectory (XY)')
ax.plot(a_out[:, 4], a_out[:, 5], '-', label='out')
ax.plot(a_pre[:, 4], a_pre[:, 5], '--', label='pre')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()
ax.legend()

# --- 3D trajectory ---
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
fig3d.suptitle('Estimated Trajectory (3D)')
ax3d.plot(a_out[:, 4], a_out[:, 5], a_out[:, 6], '-', label='out', color='y', lw=0.8)
ax3d.plot(a_pre[:, 4], a_pre[:, 5], a_pre[:, 6], '--', label='pre', color='m', lw=0.8)
ax3d.set_xlabel('x [m]')
ax3d.set_ylabel('y [m]')
ax3d.set_zlabel('z [m]')
ax3d.legend()
ax3d.grid()
pad = 0.5
ax3d.set_xlim(a_out[:, 4].min() - pad, a_out[:, 4].max() + pad)
ax3d.set_ylim(a_out[:, 5].min() - pad, a_out[:, 5].max() + pad)
ax3d.set_zlim(a_out[:, 6].min() - pad, a_out[:, 6].max() + pad)

# --- Velocity (j=4) ---
fig, ax = plt.subplots()
fig.suptitle('Velocity')
for i in range(1, 4):
    ax.plot(time, a_pre[:, i+4*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+4*3], '-', label=lab_out[i])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Velocity [m/s]')
ax.grid()
ax.legend()

# --- bg (j=5) ---
fig, ax = plt.subplots()
fig.suptitle('Bias of Gyroscope')
for i in range(1, 4):
    ax.plot(time, a_pre[:, i+5*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+5*3], '-', label=lab_out[i])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Bias [rad/s]')
ax.grid()
ax.legend()

# --- ba (j=6) ---
fig, ax = plt.subplots()
fig.suptitle('Bias of Accelerometer')
for i in range(1, 4):
    ax.plot(time, a_pre[:, i+6*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+6*3], '-', label=lab_out[i])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Bias [m/s^2]')
ax.grid()
ax.legend()

# --- Gravity (j=7) ---
fig, ax = plt.subplots()
fig.suptitle('Estimated Gravity')
for i in range(1, 4):
    ax.plot(time, a_pre[:, i+7*3], '-', label=lab_pre[i])
    ax.plot(time, a_out[:, i+7*3], '-', label=lab_out[i])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Gravity [m/s^2]')
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
#         axes[i].legend(fontsize=8)
#         axes[i].grid()
#     axes[-1].set_xlabel('Time [s]')
#     plt.tight_layout()


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
plt.grid()

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
    # pos columns: 0..24 state fields, then 36 covariance values (row-major 6x6)
    if pos.shape[1] >= 25 + 36:
        time_pos = pos[:, 0]
        # apply same time filter used above (use end_time if defined)
        # mask_pos = (time_pos >= start_time) & (time_pos <= end_time)
        # cov_flat = pos[mask_pos, 25:25+36]
        # t_cov = time_pos[mask_pos]
        cov_flat = pos[:, 25:25+36]
        t_cov = time_pos
        if cov_flat.size > 0:
            cov = cov_flat.reshape(-1, 6, 6)
            # diagonal elements (following printed row-major order)
            diag = np.array([cov[:, i, i] for i in range(6)])
            fig2, ax2 = plt.subplots(2,3, figsize=(12,6))
            ax2 = ax2.ravel()
            labels = ['cov_x','cov_y','cov_z','cov_roll','cov_pitch','cov_yaw']
            for i in range(6):
                ax2[i].plot(t_cov, diag[i], '-', label=labels[i])
                ax2[i].set_title(labels[i])
                ax2[i].grid()
                ax2[i].legend()
            fig2.suptitle('Pose covariance diagonals')
            plt.tight_layout()
        else:
            print('No covariance data found in Log/pos_log.txt for the requested time range.')
    else:
        print('Log/pos_log.txt does not contain covariance columns (need >=61 columns per line).')

    # --- Information matrix analysis ---
    # cols: 61=n_pts, 62=mean_res, 63=cost, 64-99=H^T*H (6x6 row-major)
    if pos.shape[1] >= 61 + 3 + 36:
        t_info   = time_pos
        n_pts    = pos[:, 61]
        mean_res = pos[:, 62]
        cost     = pos[:, 63]
        info_flat = pos[:, 64:100]  # (N, 36)
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

        fig3, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig3.suptitle('IEKF Information Matrix Analysis')

        axs[0, 0].plot(t_info, n_pts)
        axs[0, 0].set_title('Effective Points')
        axs[0, 0].set_xlabel('Time [s]')
        axs[0, 0].grid()

        # axs[0, 1].plot(t_info, mean_res, label='mean residual [m]')
        axs[0, 1].plot(t_info, cost,     label='cost (sum sq res)')
        axs[0, 1].set_title('Residuals')
        axs[0, 1].set_xlabel('Time [s]')
        axs[0, 1].legend()
        axs[0, 1].grid()

        eig_labels = [f'eig_{i+1}' for i in range(6)]
        for i in range(6):
            axs[1, 0].plot(t_info, eig_vals[:, i], label=eig_labels[i])
        axs[1, 0].set_title('Info Matrix Eigenvalues (ascending)')
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 0].set_yscale('log')
        axs[1, 0].legend()
        axs[1, 0].grid()
        axs[1, 1].plot(t_info, cond)
        axs[1, 1].set_title('Condition Number (eig_min / eig_max)')
        axs[1, 1].set_xlabel('Time [s]')
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

        # --- Eigenvector heatmaps: one per xyz direction, picked at its peak dominance time ---
        # H^T*H cols: 0=pos_x, 1=pos_y, 2=pos_z, 3=rot_x, 4=rot_y, 5=rot_z
        # Reorder to rot-first (paper style): [3,4,5,0,1,2] → Rx,Ry,Rz,X,Y,Z
        reorder    = [3, 4, 5, 0, 1, 2]
        col_labels = ['Rx', 'Ry', 'Rz', 'X', 'Y', 'Z']
        xyz_labels = ['X', 'Y', 'Z']
        xyz_cols   = [0, 1, 2]   # original column indices for pos_x, pos_y, pos_z

        from matplotlib.colors import PowerNorm
        import matplotlib.colorbar as mcolorbar
        gamma = 0.35   # <1 spreads low values; raise toward 1.0 for more linear

        # 4 columns: 3 heatmaps + 1 narrow colorbar column
        fig_h, axes_h = plt.subplots(1, 4, figsize=(12, 5),
                                     gridspec_kw={'width_ratios': [4, 4, 4, 0.4]})
        fig_h.suptitle('Eigenvector heatmaps at peak X / Y / Z dominance in weakest eigenvector\n'
                        '(rows = v1..v6 ascending eigenvalue, cols = Rx Ry Rz X Y Z)', fontsize=9)

        norm = PowerNorm(gamma=gamma, vmin=0, vmax=1)

        for panel, (xyz_label, orig_col) in enumerate(zip(xyz_labels, xyz_cols)):
            peak_idx = int(np.argmax(np.abs(weakest_vec[:, orig_col])))

            mat = info_mats[peak_idx][:, reorder][reorder, :]
            vals, vecs = np.linalg.eigh(mat)
            data = np.abs(vecs.T)          # row i = eigenvector i (ascending)

            ax = axes_h[panel]
            im = ax.imshow(data, cmap='Greys_r', norm=norm, aspect='auto')

            # red separator at largest eigenvalue gap
            if vals[-1] > 1e-10:
                ratios = np.diff(np.log10(np.clip(vals, 1e-10, None)))
                gap = int(np.argmax(ratios))
                ax.axhline(gap + 0.5, color='red', lw=1.2)

            ax.set_title(f'{xyz_label} dominant  t={t_info[peak_idx]:.2f}s', fontsize=8)
            ax.set_xticks(range(6))
            ax.set_xticklabels(col_labels, fontsize=7, rotation=45)
            ax.set_yticks(range(6))
            if panel == 0:
                ax.set_yticklabels(
                    [f'v{i+1}  {vals[i]:.0f}' for i in range(6)], fontsize=7)
            else:
                ax.set_yticklabels(
                    [f'{vals[i]:.0f}' for i in range(6)], fontsize=7)

        # colorbar with tick marks at representative |component| values
        cbar = fig_h.colorbar(im, cax=axes_h[3])
        cbar.set_label('|component|', fontsize=7)
        cbar.set_ticks([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0])
        cbar.ax.tick_params(labelsize=6)

        plt.tight_layout()

    else:
        print('Log/pos_log.txt does not contain info matrix columns (need >=100 columns per line).')

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

    # --- Timing overview ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle('FAST-LIO Timing breakdown')

    axes[0].plot(t_log, total, lw=0.7, label='total')
    axes[0].axhline(np.mean(total), color='r', ls='--', lw=1,
                    label='mean %.1f ms' % np.mean(total))
    axes[0].axhline(100, color='k', ls='-', lw=0.8, label='scan period 100 ms')
    axes[0].set_ylabel('Time [ms]')
    axes[0].legend(fontsize=8)
    axes[0].grid()

    _threshold = 0.01 * total.mean()   # bands < 1% of mean total → fold into other
    _bands  = [imu_proc, fov_seg, downsample, match_time, solve, iekf_other, map_incr, other]
    _labels = ['imu proc', 'fov seg', 'downsample',
               'iekf match (NN+plane)', 'iekf solve', 'iekf other',
               'map update (incr+del)', 'other']
    _merged_other = other.copy()
    _keep_bands, _keep_labels = [], []
    for b, l in zip(_bands[:-1], _labels[:-1]):   # last entry is already 'other'
        if b.mean() < _threshold:
            _merged_other += b
        else:
            _keep_bands.append(b)
            _keep_labels.append(l)
    _keep_bands.append(_merged_other)
    _keep_labels.append('other')
    _order = np.argsort([b.mean() for b in _keep_bands])
    axes[1].stackplot(t_log,
                      *[_keep_bands[i] for i in _order],
                      labels=[_keep_labels[i] for i in _order],
                      alpha=0.8)
    axes[1].set_ylabel('Time [ms]')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(fontsize=8, loc='upper left')
    axes[1].grid()
    plt.tight_layout()

    # --- Point cloud and tree sizes ---
    scan_pts = tlog[:, 2]
    add_pts  = tlog[:, 9]
    tree_st  = tlog[:, 7]
    tree_end = tlog[:, 8]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle('Point Cloud & Tree Sizes (queue=10)')

    axes[0].plot(t_log, scan_pts, lw=0.7, label='scan points')
    axes[0].plot(t_log, add_pts,  lw=0.7, label='added to tree')
    axes[0].set_ylabel('Points')
    axes[0].legend(fontsize=8)
    axes[0].grid()

    axes[1].plot(t_log, tree_st,  lw=0.7, label='tree size (start)')
    axes[1].plot(t_log, tree_end, lw=0.7, label='tree size (end)')
    axes[1].set_ylabel('Nodes')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(fontsize=8)
    axes[1].grid()
    plt.tight_layout()

except Exception as e:
    print('Could not load fast_lio_time_log.csv:', e)

plt.tight_layout()
plt.show()
