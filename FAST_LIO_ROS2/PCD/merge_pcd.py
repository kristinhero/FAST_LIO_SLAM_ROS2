import glob
import subprocess
import os

# Path to your PCD files
path = "/home/aidit/ws_slam/src/FAST_LIO_SLAM_ROS2/FAST_LIO_ROS2/PCD/square_tank_slow"
pcd_files = sorted(glob.glob(os.path.join(path, "scans_*.pcd")))

if not pcd_files:
    print("No scan PCD files found!")
    exit(1)

print(f"Merging {len(pcd_files)} files...")

# pcl_concatenate_points_pcd writes output.pcd in the working directory
result = subprocess.run(
    ["pcl_concatenate_points_pcd"] + pcd_files,
    cwd=path,
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    exit(result.returncode)

output = os.path.join(path, "output.pcd")
merged = os.path.join(path, "merged.pcd")
if os.path.exists(output):
    os.rename(output, merged)
    print(f"Saved to {merged}")
