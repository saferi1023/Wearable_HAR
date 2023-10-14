from utils.imusim import Simulator
import matplotlib.pyplot as plt
import numpy as np

# Load joints data and corresponding quat params
file_name = "example"

joint_data_path = "./output/positions_data/"+file_name+".npy"
quat_params_path = "./output/quat_params_data/"+file_name+"_quat_params.npy"

joint_data = np.squeeze(np.load(joint_data_path))   # Shape (48, 22, 3)
quaternions = np.load(quat_params_path)             # Shape (48, 22, 4)

# Identify the left wrist joint
joint_index = 18  # Assuming 19th index is for LEFT HAND WRIST

# Extract position and quaternion data
positions = joint_data[:, joint_index, :]
quats = quaternions[:, joint_index, :]

# 3. Construct the trajectory
samples = []
dt = 1/30  # Frame Rate: 1/30 for 30fps 
samples = [(frame*dt, positions[frame], quats[frame]) for frame in range(48)] 
# Assuming uniform time sampling, and dt is the time between frames


# Save the trajectory to a csv 
rows = [np.concatenate(([t[0]], t[1], t[2])) for t in samples] #Flatten
samples_data = "./output/samples/"+file_name+"_samples.csv"
np.savetxt(samples_data, rows, delimiter=',', header='time,x,y,z,qw,qx,qy,qz')

# Load IMU Simulator
sim = Simulator(samples_data)

 # Get the  simulated accelerometer, gyroscope data
accelerometer_data = sim.accelerometer
gyroscope_data = sim.gyroscope


#### PLOT
# accelerometer_data plot
time_points = np.arange(accelerometer_data.shape[0])  # Assuming time points are just indices
plt.figure(figsize=(10, 6))

plt.plot(time_points, accelerometer_data[:, 0], label='Axis 1', marker='o')
plt.plot(time_points, accelerometer_data[:, 1], label='Axis 2', marker='o')
plt.plot(time_points, accelerometer_data[:, 2], label='Axis 3', marker='o')

plt.xlabel('Frames')
plt.ylabel('Value')
plt.title('Accelerometer_Data')
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()

# gyroscope_data plot
time_points = np.arange(gyroscope_data.shape[0])  # Assuming time points are just indices
plt.figure(figsize=(10, 6))

plt.plot(time_points, gyroscope_data[:, 0], label='Axis 1', marker='o')
plt.plot(time_points, gyroscope_data[:, 1], label='Axis 2', marker='o')
plt.plot(time_points, gyroscope_data[:, 2], label='Axis 3', marker='o')

plt.xlabel('Frames')
plt.ylabel('Value')
plt.title('Gyroscope_Data')
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()

