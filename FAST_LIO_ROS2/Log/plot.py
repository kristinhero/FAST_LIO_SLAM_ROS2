# import matplotlib
# matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt

_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Log_curved_tank_fast')


#######for ikfom
lab_pre = ['', 'pre-x', 'pre-y', 'pre-z']
lab_out = ['', 'out-x', 'out-y', 'out-z']
a_pre=np.loadtxt(os.path.join(_dir, 'mat_pre.txt'))
a_out=np.loadtxt(os.path.join(_dir, 'mat_out.txt'))
time=a_pre[:,0]

# # Filter data to plot only from 110 to 190 seconds
# start_time = 110
# end_time = 190
# # mask = (time >= start_time) & (time <= end_time)
# mask = (time >= start_time) # Plot only after 110 seconds
# a_pre = a_pre[mask]
# a_out = a_out[mask]
# time = time[mask]

# --- Attitude (j=0) ---
fig, ax = plt.subplots()
fig.suptitle('Attitude')
for i in range(1, 4):
    # pre_vals = np.where(a_pre[:, i+0*3] < -105, a_pre[:, i+0*3] + 360, a_pre[:, i+0*3])
    # out_vals = np.where(a_out[:, i+0*3] < -105, a_out[:, i+0*3] + 360, a_out[:, i+0*3])
    # ax.plot(time, pre_vals, '-', label=lab_pre[i])
    # ax.plot(time, out_vals, '-', label=lab_out[i])
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

### Calculate and plot state differences (out - pre)
state_cfg = {
    0: ('Attitude',     ['Δroll [deg]',  'Δpitch [deg]', 'Δyaw [deg]'],   True),
    1: ('Translation',  ['Δx [m]',       'Δy [m]',       'Δz [m]'],       False),
    4: ('Velocity',     ['Δvx [m/s]',    'Δvy [m/s]',    'Δvz [m/s]'],    False),
    5: ('Gyro bias',    ['Δbg_x [rad/s]','Δbg_y [rad/s]','Δbg_z [rad/s]'],False),
    6: ('Acc bias',     ['Δba_x [m/s²]', 'Δba_y [m/s²]', 'Δba_z [m/s²]'],False),
    7: ('Gravity',      ['Δgx [m/s²]',   'Δgy [m/s²]',   'Δgz [m/s²]'],  False),
}
for j, (name, ylabels, wrap) in state_cfg.items():
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f'State difference (out − pre): {name}')
    for i in range(3):
        diff = a_out[:, i+1+j*3] - a_pre[:, i+1+j*3]
        if wrap:
            diff = (diff + 180) % 360 - 180
        axes[i].plot(time, diff, '-', color=f'C{i}', label=ylabels[i])
        axes[i].set_ylabel(ylabels[i])
        axes[i].legend(fontsize=8)
        axes[i].grid()
    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()


# ### Draw IMU data
# fig, axs = plt.subplots(2)
# imu=np.loadtxt('imu.txt')
# time=imu[:,0]
# axs[0].set_title('Gyroscope')
# axs[1].set_title('Accelerameter')
# lab_1 = ['gyr-x', 'gyr-y', 'gyr-z']
# lab_2 = ['acc-x', 'acc-y', 'acc-z']
# for i in range(3):
#     # if i==1:
#     axs[0].plot(time, imu[:,i+1],'.-', label=lab_1[i])
#     axs[1].plot(time, imu[:,i+4],'.-', label=lab_2[i])
# for i in range(2):
#     # axs[i].set_xlim(386,389)
#     axs[i].grid()
#     axs[i].legend()
# plt.grid()

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
# # # print(time_se)
# # # print(a_out3[:,2])
# # plt.grid()
# # plt.savefig("time.pdf", dpi=1200)

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
    else:
        print('Log/pos_log.txt does not contain info matrix columns (need >=100 columns per line).')

except Exception as e:
    print('Could not load Log/pos_log.txt:', e)

plt.tight_layout()
plt.show()
