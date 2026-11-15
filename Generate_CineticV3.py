# -*- coding: utf-8 -*-
"""
Mobile Device Simulation - 30 Devices in 100x50x15 Area
Demonstrates different mobile device types with customized parameters
Author: Karim Bechiri
"""

from DynaEdgeSimV3 import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ===================== SIMULATION CONFIGURATION =====================
print("=" * 80)
print("MOBILE DEVICE SIMULATION - 32 DEVICES")
print("=" * 80)
output_file = "mobile_sim_32C_100R_V3.csv"

# Define three types of mobile devices with different characteristics
mobile_params = {
    # ========== Simulation Settings ==========
    'MaxTime': 101,              # 20 round
    'Device_count': 32,          # 30 mobile devices
    
    # ========== Environment Dimensions ==========
    'sWidth': 50,               # 100 meters wide
    'sHeight': 20,               # 50 meters deep
    'sDepth': 30,                # 15 meters height (multi-floor building)
    
    # ========== Master Device Position (Center of area) ==========
    'MasterDeviceX': 25,         # Center X
    'MasterDeviceY': 10,         # Center Y
    'MasterDeviceZ': 15,        # Middle height
    
    # ========== Movement Parameters (Pedestrian mobility) ==========
    'MoveSpeed': 20,              # Max 3 m/s (walking speed ~10 km/h)
    'Acceleration': 2,         # Gradual acceleration
    'Deceleration': 1.5,         # Quick deceleration
    'Direction_change_prob': 0.1,  # 8% chance to change direction
    'Stop_prob': 0.1,           # 3% chance to stop (looking at phone, waiting)
    'Start_prob': 0.05,          # 6% chance to start moving
    'Moving_prob': 0.7,          # 70% start moving initially
    
    # ========== Battery Parameters ==========
    # Range represents different phone models (old to new)
    'Battery_capacity_min': 10.0,    # Old phone: 10 Wh (~2500 mAh at 4V)
    'Battery_capacity_max': 20.0,    # New phone: 20 Wh (~5000 mAh at 4V)
    'Battery_level_min': 0.2,        # Some devices low on battery
    'Battery_level_max': 1,        # Some fully charged
    'Battery_degradation_min': 5.0,  # Brand new battery
    'Battery_degradation_max': 10,  # Heavily degraded (80% faster drain)
    
    # ========== Power Consumption Parameters ==========
    # Different phone models have different power profiles
    'Idle_power_min': 2,           # Efficient phone idle: 0.4W
    'Idle_power_max': 8,           # Power-hungry phone idle: 1.2W
    'Movement_power_factor_min': 1.8,  # Low movement power (GPS off)
    'Movement_power_factor_max': 3.5,  # High movement power (GPS + sensors)
    'CPU_power_factor_min': 2.0,     # Efficient processor
    'CPU_power_factor_max': 6.0,     # Power-hungry processor
    'Network_power_idle_min': 0.05,  # Network standby
    'Network_power_idle_max': 1.2,   # Active background sync
    'Network_power_tx_min': 2.0,     # Low transmission power (good signal)
    'Network_power_tx_max': 5.5,     # High transmission power (poor signal)
    
    # ========== CPU Parameters ==========
    'CPU_capacity_min': 0.15,        # Light usage (messaging)
    'CPU_capacity_max': 0.85,        # Heavy usage (gaming, video)
    
    # ========== Network Parameters ==========
    'TX_power_min': 5.0,            # Low transmission power (dBm) - good signal
    'TX_power_max': 10.0,            # High transmission power (dBm) - poor signal
    'Latency_min': 3.0,              # Initial latency (will be recalculated)
    'Latency_max': 10.0,
    
    # ========== Network Simulation Parameters ==========
    'Speed_of_light': 3e8,           # m/s
    'Packet_size': 1500 * 8,         # 1500 bytes typical packet
    'Bandwidth': 40e6,               # 40 MHz (LTE channel)
    'Processing_delay': 0.002,       # 2ms processing
    'Path_loss_exponent': 3.0,       # Indoor environment
    'Noise_floor': -95,              # dBm (indoor)
    
    # ========== Reliability Parameters ==========
    'Reliability_min': 0.4,          # Some unreliable devices
    'Reliability_max': 0.98,         # Mostly reliable
    'Reliability_window': 5,        # Last 10 samples for reliability
    'Offline_prob': 0.1,            # 4% chance to go offline each step
    'Reliability_alpha': 0.25,       # EWMA weight for reliability
    
    # ========== Computation & communication Parameters (for future FL use) ==========
    'Computation_power': 40,        # 5W for ML computation
    'upload_model_power': 20,        # 3W for ML computation
    'download_model_power': 2,        # 2W for ML computation
    'upload_model_time': 5,        # 3second for uploading model
    'download_model_time': 2,        # 2second for downloading model
}

# ===================== CREATE AND RUN SIMULATION =====================
print("\nInitializing environment with custom parameters...")
print(f"Area: {mobile_params['sWidth']}m × {mobile_params['sHeight']}m × {mobile_params['sDepth']}m")
print(f"Devices: {mobile_params['Device_count']}")
print(f"Time steps: {mobile_params['MaxTime']}")

# Create environment
env = Environment(params=mobile_params)

# Create devices with varied characteristics
print("\nCreating 30 mobile devices with varied characteristics...")
print("Device types:")
print("  - Old smartphones (degraded battery, high power consumption)")
print("  - Mid-range phones (moderate specs)")
print("  - New flagship phones (large battery, efficient)")

env.create_random_devices()

# Show initial device statistics
print("\nInitial Device Statistics:")
print("-" * 80)
initial_devices = env.devices
batteries = [d.battery_level for d in initial_devices]
degradations = [d.battery_degradation for d in initial_devices]
capacities = [d.battery_capacity for d in initial_devices]

print(f"Battery Levels: min={min(batteries):.1%}, max={max(batteries):.1%}, avg={np.mean(batteries):.1%}")
print(f"Battery Degradation: min={min(degradations):.2f}x, max={max(degradations):.2f}x, avg={np.mean(degradations):.2f}x")
print(f"Battery Capacity: min={min(capacities):.1f}Wh, max={max(capacities):.1f}Wh, avg={np.mean(capacities):.1f}Wh")

# Run simulation
print("\n" + "=" * 80)
print("RUNNING SIMULATION")
print("=" * 80)
env.run_simulation(print_infos=False)

# ===================== SAVE SIMULATION =====================

print(f"\nSaving simulation to {output_file}...")
env.export_simulation_data(output_file)
print(f"✓ Simulation saved successfully")

# ===================== 3D MOVEMENT VISUALIZATION =====================
print("\n" + "=" * 80)
print("VISUALIZING 3D MOVEMENT")
print("=" * 80)

# Select time steps to visualize (start, quarter, half, three-quarter, end)
time_steps_to_show = [0, 1, 2, 3, 4]

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Mobile Device Movement in 3D Space - 30 Devices', 
             fontsize=16, fontweight='bold')

for idx, time_step in enumerate(time_steps_to_show):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    
    devices = env.recorded_data[time_step]
    
    # Separate online and offline devices
    online_devices = [d for d in devices if d.online]
    offline_devices = [d for d in devices if not d.online]
    
    # Plot online devices (colored by battery level)
    if online_devices:
        x_on = [d.x for d in online_devices]
        y_on = [d.y for d in online_devices]
        z_on = [d.z for d in online_devices]
        battery_on = [d.battery_level for d in online_devices]
        
        scatter_on = ax.scatter(x_on, y_on, z_on, 
                               c=battery_on, cmap='RdYlGn',
                               s=100, alpha=0.8, 
                               vmin=0, vmax=1,
                               edgecolors='black', linewidth=0.5)
    
    # Plot offline devices (black)
    if offline_devices:
        x_off = [d.x for d in offline_devices]
        y_off = [d.y for d in offline_devices]
        z_off = [d.z for d in offline_devices]
        
        ax.scatter(x_off, y_off, z_off, 
                  c='black', s=100, alpha=0.5,
                  marker='x', linewidth=2)
    
    # Plot master device (edge server)
    ax.scatter([env.master_device.x], [env.master_device.y], [env.master_device.z],
              c='red', marker='*', s=400, 
              edgecolors='black', linewidth=1.5,
              label='Edge Server')
    
    # Set labels and limits
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_zlabel('Z (meters)', fontsize=10)
    ax.set_xlim([0, mobile_params['sWidth']])
    ax.set_ylim([0, mobile_params['sHeight']])
    ax.set_zlim([0, mobile_params['sDepth']])
    
    # Title with statistics
    online_count = len(online_devices)
    total_count = len(devices)
    ax.set_title(f'Time Step {time_step}\n{online_count}/{total_count} devices online', 
                fontsize=11, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for first plot
    if idx == 0 and online_devices:
        cbar = plt.colorbar(scatter_on, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Battery Level', fontsize=9)

# Add legend to last subplot
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('mobile_movement_3d.png', dpi=150, bbox_inches='tight')
print("✓ 3D visualization saved as 'mobile_movement_3d.png'")
plt.show()

# ===================== TRAJECTORY VISUALIZATION =====================
print("\nGenerating trajectory visualization...")

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories for first 10 devices (to avoid clutter)
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for device_id in range(1, min(11, mobile_params['Device_count'] + 1)):
    # Extract trajectory
    trajectory_x = []
    trajectory_y = []
    trajectory_z = []
    
    for time_step in range(mobile_params['MaxTime']):
        for device in env.recorded_data[time_step]:
            if device.device_id == device_id and device.online:
                trajectory_x.append(device.x)
                trajectory_y.append(device.y)
                trajectory_z.append(device.z)
                break
    
    if trajectory_x:
        # Plot trajectory line
        ax.plot(trajectory_x, trajectory_y, trajectory_z, 
               color=colors[device_id - 1], alpha=0.6, linewidth=1.5,
               label=f'Device {device_id}')
        
        # Mark start and end points
        ax.scatter([trajectory_x[0]], [trajectory_y[0]], [trajectory_z[0]], 
                  color=colors[device_id - 1], s=100, marker='o', 
                  edgecolors='black', linewidth=1)
        ax.scatter([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]], 
                  color=colors[device_id - 1], s=100, marker='s',
                  edgecolors='black', linewidth=1)

# Plot master device
ax.scatter([env.master_device.x], [env.master_device.y], [env.master_device.z],
          c='red', marker='*', s=500, 
          edgecolors='black', linewidth=2,
          label='Edge Server', zorder=100)

ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (meters)', fontsize=12, fontweight='bold')
ax.set_xlim([0, mobile_params['sWidth']])
ax.set_ylim([0, mobile_params['sHeight']])
ax.set_zlim([0, mobile_params['sDepth']])

ax.set_title('Device Trajectories (First 10 Devices)\nCircle=Start, Square=End', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mobile_trajectories.png', dpi=150, bbox_inches='tight')
print("✓ Trajectory visualization saved as 'mobile_trajectories.png'")
plt.show()

# ===================== FINAL STATISTICS =====================
print("\n" + "=" * 80)
print("FINAL SIMULATION STATISTICS")
print("=" * 80)

final_devices = env.recorded_data[mobile_params['MaxTime'] - 1]
final_online = [d for d in final_devices if d.online]
final_offline = [d for d in final_devices if not d.online]

print(f"\nFinal Status (Time Step {mobile_params['MaxTime'] - 1}):")
print(f"  Online devices:  {len(final_online)}/{len(final_devices)} ({len(final_online)/len(final_devices)*100:.1f}%)")
print(f"  Offline devices: {len(final_offline)}/{len(final_devices)} ({len(final_offline)/len(final_devices)*100:.1f}%)")

if final_online:
    final_batteries = [d.battery_level for d in final_online]
    final_latencies = [d.latency for d in final_online]
    final_reliability = [d.reliability for d in final_online]
    
    print(f"\nOnline Device Statistics:")
    print(f"  Battery Level:")
    print(f"    Min:  {min(final_batteries):.1%}")
    print(f"    Max:  {max(final_batteries):.1%}")
    print(f"    Avg:  {np.mean(final_batteries):.1%}")
    print(f"    Std:  {np.std(final_batteries):.1%}")
    
    print(f"  Latency (ms):")
    print(f"    Min:  {min(final_latencies):.2f}")
    print(f"    Max:  {max(final_latencies):.2f}")
    print(f"    Avg:  {np.mean(final_latencies):.2f}")
    
    print(f"  Reliability:")
    print(f"    Min:  {min(final_reliability):.1%}")
    print(f"    Max:  {max(final_reliability):.1%}")
    print(f"    Avg:  {np.mean(final_reliability):.1%}")

# Count devices by battery degradation level
low_deg = [d for d in final_devices if d.battery_degradation < 1.2]
mid_deg = [d for d in final_devices if 1.2 <= d.battery_degradation < 1.5]
high_deg = [d for d in final_devices if d.battery_degradation >= 1.5]

print(f"\nDevice Age Distribution:")
print(f"  New devices (degradation < 1.2x):  {len(low_deg)} devices")
print(f"  Mid-age (1.2x ≤ degradation < 1.5x): {len(mid_deg)} devices")
print(f"  Old devices (degradation ≥ 1.5x):   {len(high_deg)} devices")

print("\n" + "=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  1. {output_file} - Simulation data (can be loaded for Real-Time mode)")
print(f"  2. mobile_movement_3d.png - 3D snapshots at different time steps")
print(f"  3. mobile_trajectories.png - Movement trajectories for first 10 devices")
print("\nYou can now load this simulation in Real-Time mode for Federated Learning!")
print("=" * 80)