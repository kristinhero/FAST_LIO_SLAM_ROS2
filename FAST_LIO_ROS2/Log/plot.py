# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


#######for ikfom
fig, axs = plt.subplots(4,2)
plt.subplots_adjust(hspace=0.35)
lab_pre = ['', 'pre-x', 'pre-y', 'pre-z']
lab_out = ['', 'out-x', 'out-y', 'out-z']
plot_ind = range(7,10)
# a_pre=np.loadtxt('Log_failure/mat_pre_ext_enable_fail.txt')
# a_out=np.loadtxt('Log_failure/mat_out_ext_enable_fail.txt')
a_pre=np.loadtxt('mat_pre.txt')
a_out=np.loadtxt('mat_out.txt')
time=a_pre[:,0]

# Filter data to plot only from 110 to 190 seconds
start_time = 110
end_time = 190
# mask = (time >= start_time) & (time <= end_time)
mask = (time >= start_time) # Plot only after 110 seconds
a_pre = a_pre[mask]
a_out = a_out[mask]
time = time[mask]
axs[0,0].set_title('Attitude')
axs[1,0].set_title('Translation')
axs[2,0].set_title('Extrins-R')
axs[3,0].set_title('Extrins-T')
axs[0,1].set_title('Velocity')
axs[1,1].set_title('bg')
axs[2,1].set_title('ba')
axs[3,1].set_title('Gravity')
for i in range(1,4):
    for j in range(8):
        axs[j%4, j//4].plot(time, a_pre[:,i+j*3],'-', label=lab_pre[i])
        axs[j%4, j//4].plot(time, a_out[:,i+j*3],'-', label=lab_out[i])
for j in range(8):
    # axs[j].set_xlim(386,389)
    axs[j%4, j//4].grid()
    axs[j%4, j//4].legend()
plt.grid()
#######for ikfom#######


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
    pos = np.loadtxt('pos_log.txt')
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    # pos columns: 0..24 state fields, then 36 covariance values (row-major 6x6)
    if pos.shape[1] >= 25 + 36:
        time_pos = pos[:, 0]
        # apply same time filter used above (use end_time if defined)
        # mask_pos = (time_pos >= start_time) & (time_pos <= end_time)
        mask_pos = (time_pos >= start_time)

        cov_flat = pos[mask_pos, 25:25+36]
        if cov_flat.size > 0:
            cov = cov_flat.reshape(-1, 6, 6)
            # diagonal elements (following printed row-major order)
            diag = np.array([cov[:, i, i] for i in range(6)])
            fig2, ax2 = plt.subplots(2,3, figsize=(12,6))
            ax2 = ax2.ravel()
            labels = ['cov_x','cov_y','cov_z','cov_roll','cov_pitch','cov_yaw']
            t_cov = time_pos[mask_pos]
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
except Exception as e:
    print('Could not load Log/pos_log.txt:', e)

plt.tight_layout()
plt.show()
