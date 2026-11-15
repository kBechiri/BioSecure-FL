# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 00:10:53 2025
DynaEdgeSim (Dynamic Edge Device Simulator) - Enhanced Realistic Version
@author: karim bechiri 

Copyright (c) 2025 karim Bechiri

This software is provided free of charge for educational and non-commercial use only.
For commercial licensing options, 
please contact the author at karim.bechiri@university-constantine2.dz.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import random
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from copy import copy
from collections import deque
import os
import csv

def ru(vmin, vmax):
    return random.uniform(vmin, vmax)

# Each device is represented as an object with fixed and variable properties.
class Device:
    def __init__(self, device_id: int, x, y, z, maxspeed, acceleration, deceleration,
                 direction_change_prob, stop_prob, start_prob, cpu_capacity, 
                 battery_capacity, battery_level: float, battery_degradation: float,
                 idle_power: float, movement_power_factor: float, cpu_power_factor: float,
                 network_power_idle: float, network_power_tx: float,
                 latency: float, reliability: float, reliability_window: float, 
                 tx_power: float, state=True):
        # Fixed properties
        self.device_id = device_id        
        
        # Position
        self.x = x
        self.y = y
        self.z = z
        
        # Computing resources
        self.cpu_capacity = cpu_capacity
        self.active_tasks = []  # List of tasks being processed
        
        # Battery properties (more realistic with aging)
        self.battery_capacity = battery_capacity  # Total capacity in Wh
        self.battery_level = battery_level  # Current level as percentage
        self.battery_wh = battery_capacity * battery_level  # Actual energy in Wh
        self.battery_degradation = battery_degradation  # Degradation factor (1.0 = new, >1.0 = aged)
        
        # Power consumption parameters (device-specific based on age/model)
        self.idle_power = idle_power  # Base idle power (W)
        self.movement_power_factor = movement_power_factor  # Movement power multiplier
        self.cpu_power_factor = cpu_power_factor  # CPU power multiplier
        self.network_power_idle = network_power_idle  # Network idle power (W)
        self.network_power_tx = network_power_tx  # Network transmission power (W)
        
        # Network properties
        self.latency = latency
        self.reliability = reliability
        self.tx_power = tx_power  # Transmission power in dBm
        self.transmitting = False
        self.online = state
        
        # Reliability tracking
        self.state_history = deque(maxlen=reliability_window)
        self.state_history.append(state)
        self.total_uptime = 0
        self.total_downtime = 0
        
        # Real-Time simulation control (for federated learning)
        self.is_selected = False  # Whether device is selected for computation
        self.computation_power = 0.0  # Additional power for ML computation (W)
        
        # Movement properties
        self.direction = random.uniform(0, 2 * np.pi)
        self.speed = 1
        self.max_speed = ru(2, maxspeed)
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.direction_change_prob = direction_change_prob
        self.stop_prob = stop_prob
        self.start_prob = start_prob
        self.is_moving = False
        self.stationary_time = 0
        self.moving_time = 0


    def update_location(self, new_x, new_y, new_z, env_width, env_height, env_depth):
        if self.online:
            buffer = 1.0
            self.x = max(buffer, min(new_x, env_width - buffer))
            self.y = max(buffer, min(new_y, env_height - buffer))
            self.z = max(buffer, min(new_z, env_depth - buffer))            

    def update_cpu_capacity(self, base_load=0.1, network_overhead=0.05):
        """More realistic CPU usage model"""
        if self.online:
            # Base OS overhead
            task_load = sum(min(0.3, ru(0.1, 0.4)) for _ in self.active_tasks) if self.active_tasks else 0
            net_load = network_overhead if self.transmitting else 0
            
            # Add some natural variation
            noise = ru(-0.05, 0.05)
            self.cpu_capacity = min(1.0, max(0.0, base_load + task_load + net_load + noise))
            
    def update_battery_realistic(self, time_step_seconds=1.0, computation_power_extra=0.0):
        """
        More realistic battery drain model based on power consumption
        Power components: idle, movement, CPU, network transmission, computation
        Battery degradation factor affects overall consumption
        
        computation_power_extra: Additional power for ML/FL computation (W)
        """
        if self.online:
            # Movement power (quadratic relationship with speed)
            movement_power = self.movement_power_factor * (self.speed / self.max_speed) ** 2 if self.is_moving else 0
            
            # CPU power (linear with usage)
            cpu_power = self.cpu_power_factor * self.cpu_capacity
            
            # Network transmission power
            network_power = self.network_power_tx if self.transmitting else self.network_power_idle
            
            # Computation power (if device is selected for federated learning)
            comp_power = computation_power_extra if self.is_selected else 0.0
            
            # Total power consumption with degradation factor
            # Aged batteries drain faster due to increased internal resistance
            total_power = (self.idle_power + movement_power + cpu_power + network_power + comp_power) * self.battery_degradation
            
            # Energy consumed in this time step (Wh)
            energy_consumed = total_power * (time_step_seconds / 3600.0)
            
            # Update battery
            self.battery_wh -= energy_consumed
            self.battery_level = max(0.0, min(1.0, self.battery_wh / self.battery_capacity))
            
            # Clamp battery_wh to valid range [0, capacity]
            self.battery_wh = max(0.0, min(self.battery_wh, self.battery_capacity))
            
            # Device shuts down if battery is critically low
            if self.battery_level <= 0.05:
                self.online = False
                self.battery_level = 0.03
                self.is_selected = False  # Deselect if goes offline
        else:
            # Device is charging/offline - recharge to random level (cannot exceed capacity)
            self.battery_wh = min(self.battery_capacity * ru(0.8, 1.0), self.battery_capacity)
            self.battery_level = min(1.0, self.battery_wh / self.battery_capacity)
            self.online = True
        
    def update_latency_realistic(self, distance_to_master, nearby_devices, 
                                 speed_of_light=3e8, packet_size=12000, bandwidth=20e6,
                                 processing_delay=0.001, path_loss_exponent=3.5,
                                 noise_floor=-90):
        """
        More realistic latency calculation with wireless channel model
        Uses log-distance path loss model and queuing theory
        All parameters can be customized
        """
        if self.online:
            # 1. Propagation delay (distance/speed)
            propagation_delay = distance_to_master / speed_of_light
            
            # 2. Transmission delay (packet_size/bandwidth)
            transmission_delay = packet_size / bandwidth
            
            # 3. Queuing delay (non-linear with congestion)
            # Using M/M/1 queue approximation
            arrival_rate = len(nearby_devices) * 10  # packets per second
            service_rate = bandwidth / packet_size
            if arrival_rate < service_rate:
                utilization = arrival_rate / service_rate
                queuing_delay = utilization / (service_rate * (1 - utilization))
            else:
                queuing_delay = 0.1  # Network saturated
            
            # 4. Calculate Signal-to-Noise Ratio (SNR) for packet loss
            # Log-distance path loss model
            reference_distance = 1.0  # meters
            
            # Path loss in dB
            if distance_to_master > 0:
                path_loss = 10 * path_loss_exponent * np.log10(distance_to_master / reference_distance)
            else:
                path_loss = 0
            
            # Received signal strength
            signal_strength = self.tx_power - path_loss
            
            # Interference from nearby devices
            interference = sum(5.0 for _ in nearby_devices if _ > 0)  # Simplified
            total_noise = 10 * np.log10(10**(noise_floor/10) + 10**(interference/10))
            
            # SNR in dB
            snr = signal_strength - total_noise
            
            # Packet success rate based on SNR (simplified BER model)
            if snr > 20:
                success_rate = 0.99
            elif snr > 10:
                success_rate = 0.95
            elif snr > 5:
                success_rate = 0.85
            else:
                success_rate = 0.7
            
            # Retransmission delay if packet fails
            retransmission_delay = 0.0
            if random.random() > success_rate:
                retransmission_delay = transmission_delay * 2  # Retransmit
            
            # Total latency in seconds
            total_latency = (propagation_delay + transmission_delay + 
                           queuing_delay + processing_delay + retransmission_delay)
            
            # Add small random variation (jitter)
            jitter = ru(0.95, 1.05)
            self.latency = total_latency * jitter * 1000  # Convert to ms
        
    def update_reliability_realistic(self, alpha=0.3):
        """
        Realistic reliability calculation using availability metric
        Reliability = Uptime / (Uptime + Downtime)
        alpha: weight for exponential moving average (0-1)
        """
        # Update state history
        self.state_history.append(self.online)
        
        # Track uptime/downtime
        if self.online:
            self.total_uptime += 1
        else:
            self.total_downtime += 1
        
        # Calculate availability-based reliability
        if len(self.state_history) > 0:
            online_count = sum(1 for state in self.state_history if state)
            window_reliability = online_count / len(self.state_history)
        else:
            window_reliability = 1.0
        
        # Apply exponential weighted moving average (recent history weighted more)
        current_state_value = 1.0 if self.online else 0.0
        self.reliability = alpha * current_state_value + (1 - alpha) * window_reliability

    def __str__(self):
        return (f"Device {self.device_id}: "
                f"Pos=({self.x:5.2f}, {self.y:5.2f}, {self.z:5.2f})- "
                f"Cpu= {self.cpu_capacity:.2%} - "
                f"Bat= {self.battery_level:.2%} ({self.battery_wh:.2f}Wh) - "
                f"Deg= {self.battery_degradation:.2f}x - "
                f"Lat= {self.latency:.2f}ms - "
                f"Rel= {self.reliability:.2%} - "
                f"On= {self.online}")


# The MasterDevice (Edge) is a special device with ID=0
class MasterDevice(Device):
    def __init__(self, x: float, y: float, z: float):
        super().__init__(device_id=0, x=x, y=y, z=z, maxspeed=0,
                         acceleration=0, deceleration=0, direction_change_prob=0, 
                         stop_prob=0, start_prob=0, cpu_capacity=float('inf'), 
                         battery_capacity=float('inf'), battery_level=1.0, 
                         battery_degradation=1.0,
                         idle_power=0, movement_power_factor=0, cpu_power_factor=0,
                         network_power_idle=0, network_power_tx=0,
                         latency=0, reliability=1.0, reliability_window=1, 
                         tx_power=30.0, state=True)


# The environment manages all devices and runs the simulation.
class Environment:
    def __init__(self, params=None):
        """
        Initialize the environment with customizable parameters
        
        params: Dictionary containing simulation parameters
        """
        # Default parameters
        default_params = {
            # Simulation settings
            'MaxTime': 100,
            'Device_count': 20,
            
            # Environment dimensions
            'sWidth': 100,
            'sHeight': 100,
            'sDepth': 50,
            
            # Master device position
            'MasterDeviceX': 50,
            'MasterDeviceY': 50,
            'MasterDeviceZ': 25,
            
            # Movement parameters
            'MoveSpeed': 5,
            'Acceleration': 0.5,
            'Deceleration': 0.3,
            'Direction_change_prob': 0.05,
            'Stop_prob': 0.02,
            'Start_prob': 0.05,
            'Moving_prob': 0.6,
            
            # Battery parameters (for device creation)
            'Battery_capacity_min': 8.0,
            'Battery_capacity_max': 12.0,
            'Battery_level_min': 0.3,  # Initial battery level range
            'Battery_level_max': 1.0,
            'Battery_degradation_min': 1.0,  # New device
            'Battery_degradation_max': 1.5,  # Aged device (50% faster drain)
            
            # Power consumption parameters (device-specific ranges)
            'Idle_power_min': 0.3,
            'Idle_power_max': 0.7,
            'Movement_power_factor_min': 1.0,
            'Movement_power_factor_max': 2.0,
            'CPU_power_factor_min': 1.5,
            'CPU_power_factor_max': 2.5,
            'Network_power_idle_min': 0.05,
            'Network_power_idle_max': 0.15,
            'Network_power_tx_min': 0.8,
            'Network_power_tx_max': 1.5,
            
            # CPU parameters (initial values for device creation)
            'CPU_capacity_min': 0.1,
            'CPU_capacity_max': 1.0,
            
            # Network parameters (for device creation)
            'TX_power_min': 15.0,  # dBm
            'TX_power_max': 23.0,  # dBm
            'Latency_min': 0.6,  # Initial latency range (will be recalculated)
            'Latency_max': 0.9,
            
            # Network simulation parameters
            'Speed_of_light': 3e8,  # m/s
            'Packet_size': 12000,  # bits
            'Bandwidth': 20e6,  # Hz (20 MHz)
            'Processing_delay': 0.001,  # seconds
            'Path_loss_exponent': 3.5,  # Urban environment
            'Noise_floor': -90,  # dBm
            
            # Reliability parameters (for device creation)
            'Reliability_min': 0.75,
            'Reliability_max': 1.0,
            'Reliability_window': 50,
            'Offline_prob': 0.02,
            'Reliability_alpha': 0.3,  # EWMA weight
            
            # Computation parameters (for Real-Time mode with federated learning)
            'Computation_power': 3.0,  # Additional power for ML computation (W)
            'upload_model_power': 2,          # W (uplink)
            'download_model_power': 2,        # W (downlink)
        }
        
        # Merge with user-provided parameters
        if params is None:
            self.parameters = default_params
        else:
            self.parameters = {**default_params, **params}
        
        self.devices = []
        self.master_device = MasterDevice(
            x=self.parameters['MasterDeviceX'],
            y=self.parameters['MasterDeviceY'],
            z=self.parameters['MasterDeviceZ']
        )
                
        self.recorded_data: Dict[int, List[Dict]] = {}
        
        # Real-Time simulation mode control
        self.simulation_mode = 'STATIC'  # 'STATIC' or 'REALTIME'
        self.current_time_step = 0
        self.max_time_step = 0
        self.rt_devices = []  # Devices in real-time mode
        
    def add_device(self, device: Device):
        self.devices.append(device)

    def set_master_device(self, master_device: MasterDevice):
        self.master_device = master_device
    
    def GetDistance(self, d1, d2=None):
        if d2 is None:
            d2 = self.master_device
        return ((d1.x - d2.x) ** 2 +
                (d1.y - d2.y) ** 2 +
                (d1.z - d2.z) ** 2) ** 0.5

    def update_device_movement(self, device):
        """Movement update with realistic physics"""
        if device.is_moving:
            device.moving_time += 1
            stop_chance = device.stop_prob * (1 + device.moving_time / 20)
            if random.random() < min(stop_chance, 0.1):
                device.is_moving = False
                device.stationary_time = 0
                device.speed = max(0, device.speed - device.deceleration)
        else:
            device.stationary_time += 1
            start_chance = device.start_prob * (1 + device.stationary_time / 10)
            if random.random() < min(start_chance, 0.6):
                device.is_moving = True
                device.moving_time = 0
                
        # Direction changes
        if device.is_moving and random.random() < device.direction_change_prob:
            device.direction += random.uniform(-0.5, 0.5)
            
        # Update speed with inertia
        if device.is_moving:
            device.speed = min(device.speed + device.acceleration, device.max_speed)
        else:
            device.speed = max(0, device.speed - device.deceleration)
            
        # Calculate new position
        if device.speed > 0:
            new_x = device.x + np.cos(device.direction) * device.speed
            new_y = device.y + np.sin(device.direction) * device.speed
            new_z = device.z + device.speed * ru(-0.1, 0.1)
            
            # Boundary collision with reflection
            buffer = 1.0
            if new_x <= buffer or new_x >= self.parameters['sWidth'] - buffer:
                device.direction = np.pi - device.direction
                new_x = max(buffer, min(new_x, self.parameters['sWidth'] - buffer))
            
            if new_y <= buffer or new_y >= self.parameters['sHeight'] - buffer:
                device.direction = 2 * np.pi - device.direction
                new_y = max(buffer, min(new_y, self.parameters['sHeight'] - buffer))
                
            if new_z <= buffer or new_z >= self.parameters['sDepth'] - buffer:
                new_z = max(buffer, min(new_z, self.parameters['sDepth'] - buffer))
                
            device.update_location(new_x, new_y, new_z, self.parameters['sWidth'], 
                                 self.parameters['sHeight'], self.parameters['sDepth'])
    
    def update_device_cpu_capacity(self, device):
        """Update CPU with realistic task-based model"""
        device.update_cpu_capacity()
        
    def update_device_battery(self, device):    
        """Update battery using realistic power consumption model"""
        # Get computation power if in real-time mode
        comp_power = self.parameters.get('Computation_power', 3.0) if self.simulation_mode == 'REALTIME' else 0.0
        device.update_battery_realistic(time_step_seconds=1.0, computation_power_extra=comp_power)
    
    def update_device_latency(self, device):
        """Update latency using wireless channel model with custom parameters"""
        distance = self.GetDistance(device)
        
        # Count nearby devices for interference
        nearby_devices = []
        for d in self.devices:
            if d.online and d.device_id != device.device_id:
                dist = self.GetDistance(device, d)
                if dist < 10:  # Within 10 units
                    nearby_devices.append(dist)
        
        # Use environment parameters for latency calculation
        device.update_latency_realistic(
            distance, 
            nearby_devices,
            speed_of_light=self.parameters['Speed_of_light'],
            packet_size=self.parameters['Packet_size'],
            bandwidth=self.parameters['Bandwidth'],
            processing_delay=self.parameters['Processing_delay'],
            path_loss_exponent=self.parameters['Path_loss_exponent'],
            noise_floor=self.parameters['Noise_floor']
        )
        
    def update_device_reliability(self, device): 
        """Update reliability with random offline events"""
        if random.random() < self.parameters['Offline_prob']:
            device.online = False
        else:
            if random.random() < 0.8:
                device.online = True
                if device.battery_level < 0.05:
                    device.battery_wh = device.battery_capacity * ru(0.75, 1.0)
                    device.battery_level = device.battery_wh / device.battery_capacity
        
        device.update_reliability_realistic(alpha=self.parameters['Reliability_alpha'])
        
    def update_devices(self, time_step: int, print_infos: bool, device_id={}):
        for device in self.devices:
            if device.online:
                self.update_device_movement(device)        
                self.update_device_latency(device)
                self.update_device_battery(device)                        
                self.update_device_cpu_capacity(device)
                 
            self.update_device_reliability(device)        
            
            # Record device data
            if time_step not in self.recorded_data:
                self.recorded_data[time_step] = []
            self.recorded_data[time_step].append(copy(device))
            
            if print_infos:
                if device_id == {}:
                    print(device) 
                else:
                    if device.device_id in device_id:
                        print(device) 
            
    def run_simulation(self, print_infos: bool = False, device_id={}):
        """Run the simulation for MaxTime steps"""
        for time_step in range(self.parameters['MaxTime']):
            self.update_devices(time_step, print_infos, device_id)
        print(f"Simulation completed: {self.parameters['MaxTime']} time steps")

    def GetActiveDevices(self, time_step: int, device_id={}):
        activedevice = []
        if time_step not in self.recorded_data:
            return activedevice        
        for d in self.recorded_data[time_step]:
            if d.online:
                if device_id == {}:
                    activedevice.append(d)
                else:
                    if d.device_id in device_id:
                        activedevice.append(d)
        return activedevice

    def PlotSimulationDevices(self, time_step: int, device_ids={}, online_only: bool = False, Export2Gif: bool = False):
        if time_step not in self.recorded_data:
            print(f"No data recorded for time step {time_step}.")
            return
        else:
            print("plotting time:", time_step)
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        all_devices = self.get_devices_at_time(time_step, online_only=online_only, device_ids=device_ids)
        
        cmap = cm.RdYlGn
        
        if all_devices:
            x_vals = [d.x for d in all_devices]
            y_vals = [d.y for d in all_devices]
            z_vals = [d.z for d in all_devices]
            battery_levels = [d.battery_level for d in all_devices]
            all_ids = [d.device_id for d in all_devices]
            online_status = [d.online for d in all_devices]
            
            colors = []
            for battery, online in zip(battery_levels, online_status):
                if not online:
                    colors.append(0)
                else:
                    colors.append(battery)
            
            c = len(all_ids)
            fsize = np.select([c <= 20, c <= 50, c > 50], [16, 12, 10])  
            marker_size = fsize * 7
             
            scatter = ax.scatter(
                x_vals, y_vals, z_vals, 
                c=colors, cmap=cmap,
                marker='o', s=marker_size,
                vmin=0, vmax=1
            )
           
            for x, y, z, dev_id in zip(x_vals, y_vals, z_vals, all_ids):
                ax.text(x+2, y, z, str(dev_id), fontsize=fsize)
        
        if self.master_device:
            ax.scatter([self.master_device.x], [self.master_device.y], [self.master_device.z], 
                      c='black', marker='*', s=200, label='Master Device')
            ax.text(self.master_device.x+2, self.master_device.y, self.master_device.z, 
                   "Master", fontsize=12, color='black')
        
        ax.set_xlim([0, self.parameters['sWidth']])
        ax.set_ylim([0, self.parameters['sHeight']])
        ax.set_zlim([0, self.parameters['sDepth']])
        
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title(f"Device Positions at Time {time_step}\nTotal Devices: {len(all_devices)}")
        
        if all_devices:
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Battery Level (Online) / black=Offline')
        
        online_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                 markersize=10, label='Online Devices')
        offline_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                                  markersize=10, label='Offline Devices')
        master_patch = plt.Line2D([0], [0], marker='*', color='black', 
                                 markersize=10, label='Master Device')
        
        ax.legend(handles=[online_patch, offline_patch, master_patch], 
                 loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
    
        if Export2Gif:
            os.makedirs("animation_frames", exist_ok=True)
            frame_path = f"animation_frames/frame_{time_step:04d}.png"
            try:
                plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                plt.close()
                return frame_path
            except Exception as e:
                print(f"Error saving frame {time_step}: {str(e)}")
                plt.close()
                return None
        else:
            plt.show()
            return None

    def create_animation(self, filename="animation.gif", fps=10, device_ids={}, online_only=False):
        """Creates a GIF animation of the simulation"""
        from PIL import Image
        import shutil
        
        if os.path.exists("animation_frames"):
            try:
                shutil.rmtree("animation_frames")
            except Exception as e:
                print(f"Warning: Could not clean up old frames: {str(e)}")
        
        frame_paths = []
        for time_step in range(self.parameters['MaxTime']):
            frame_path = self.PlotSimulationDevices(
                time_step, 
                device_ids=device_ids,
                online_only=online_only,
                Export2Gif=True
            )
            if frame_path:
                frame_paths.append(frame_path)
        
        if not frame_paths:
            print("Error: No frames were generated for animation")
            return
        
        try:
            frame_paths = sorted(frame_paths)
            frames = []
            for frame_path in frame_paths:
                try:
                    frames.append(Image.open(frame_path))
                except Exception as e:
                    print(f"Error loading frame {frame_path}: {str(e)}")
            
            if not frames:
                print("Error: No valid frames found for animation")
                return
            
            frames[0].save(
                filename,
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=1000//fps,
                loop=0,
                optimize=True
            )
            print(f"Successfully created animation: {filename}")
            
        except Exception as e:
            print(f"Error creating animation: {str(e)}")
        
        try:
            shutil.rmtree("animation_frames")
        except Exception as e:
            print(f"Warning: Could not clean up frames: {str(e)}")
            
    def PrintSimulationDevices(self, time_step: int, device_id={}):
        if time_step not in self.recorded_data:
            print(f"No data recorded for time step {time_step}.")
            return
        else:
            print("Time:", time_step)        
        active_devices = self.GetActiveDevices(time_step, device_id)
        if not active_devices:
            print(f"No active devices at time {time_step}.")
            return
        for d in active_devices:
            print(d)
         
    def export_simulation_data(self, filename: str) -> None:
        """
        Export simulation data to CSV file
        Exports only fields that change during simulation or are device-specific constants
        Movement parameters are not exported as they're in the PARAMETERS section
        """
        if not self.recorded_data:
            raise ValueError("No simulation data to export")
        
        try:
            fieldnames = [
                'Timestamp', 'device_id', 
                # Position (changes)
                'x', 'y', 'z', 
                # Resources (change)
                'cpu_capacity', 'battery_level', 'battery_wh',
                # Network (change)
                'latency', 'reliability',
                # State (changes)
                'online', 'is_moving', 'speed',
                # Device-specific constants (for Real-Time mode battery calculations)
                'battery_capacity', 'battery_degradation',
                'idle_power', 'movement_power_factor', 'cpu_power_factor',
                'network_power_idle', 'network_power_tx', 'tx_power'
            ]
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                writer.writerow(['[PARAMETERS]'])
                for param_name, param_value in self.parameters.items():
                    writer.writerow([param_name, param_value])
                    
                writer.writerow([])
                                               
                writer.writerow(['[DATA]'])
                data_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                data_writer.writeheader()
                
                for timestamp, devices in self.recorded_data.items():
                    for device in devices:
                        row = {
                            'Timestamp': timestamp,
                            'device_id': device.device_id,
                            # Position
                            'x': round(device.x, 4),
                            'y': round(device.y, 4),
                            'z': round(device.z, 4),
                            # Resources
                            'cpu_capacity': round(device.cpu_capacity, 4),
                            'battery_level': round(device.battery_level, 4),
                            'battery_wh': round(device.battery_wh, 4),
                            # Network
                            'latency': round(device.latency, 4),
                            'reliability': round(device.reliability, 4),
                            # State
                            'online': device.online,
                            'is_moving': device.is_moving,
                            'speed': round(device.speed, 4),
                            # Device-specific constants
                            'battery_capacity': round(device.battery_capacity, 4),
                            'battery_degradation': round(device.battery_degradation, 4),
                            'idle_power': round(device.idle_power, 4),
                            'movement_power_factor': round(device.movement_power_factor, 4),
                            'cpu_power_factor': round(device.cpu_power_factor, 4),
                            'network_power_idle': round(device.network_power_idle, 4),
                            'network_power_tx': round(device.network_power_tx, 4),
                            'tx_power': round(device.tx_power, 4),
                        }
                        data_writer.writerow(row)
                        
            print(f"Successfully exported simulation data to {filename}")
            
        except Exception as e:
            print(f"Error exporting simulation data: {str(e)}")
            raise

    def load_simulation_data(self, filename: str) -> Dict[int, List[Device]]:
        """
        Load simulation data from CSV file
        Reconstructs devices with data from CSV and movement parameters from PARAMETERS section
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"CSV file not found: {filename}")
            
        recorded_data = {}
        current_section = None
        
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                
                for row in reader:
                    if not row:
                        continue
                    
                    if row[0] == '[PARAMETERS]':
                        current_section = 'PARAMETERS'
                        continue
                   
                    elif row[0] == '[DATA]':
                        current_section = 'DATA'
                        next(reader)  # Skip header row
                        continue
                    
                    if current_section == 'PARAMETERS':
                        param_name = row[0]
                        param_value = float(row[1]) if '.' in row[1] else int(row[1])
                        self.parameters[param_name] = param_value                        
                        continue
                    
                    elif current_section == 'DATA':
                        # Current CSV format (from the provided data):
                        # 0:Timestamp, 1:device_id, 2:x, 3:y, 4:z,
                        # 5:cpu_capacity, 6:battery_level, 7:battery_wh, 8:latency,
                        # 9:reliability, 10:online, 11:is_moving, 12:speed,
                        # 13:battery_capacity, 14:battery_degradation, 15:idle_power,
                        # 16:movement_power_factor, 17:cpu_power_factor,
                        # 18:network_power_idle, 19:network_power_tx, 20:tx_power
                        # We'll use defaults from environment parameters
                        
                        device = Device(
                            device_id=int(row[1]),
                            # Position
                            x=float(row[2]),
                            y=float(row[3]),
                            z=float(row[4]),
                            # Movement parameters from PARAMETERS (not per-device)
                            maxspeed=self.parameters.get('MoveSpeed', 5),
                            acceleration=self.parameters.get('Acceleration', 0.5),
                            deceleration=self.parameters.get('Deceleration', 0.3),
                            direction_change_prob=self.parameters.get('Direction_change_prob', 0.05),
                            stop_prob=self.parameters.get('Stop_prob', 0.02),
                            start_prob=self.parameters.get('Start_prob', 0.05),
                            # Resources
                            cpu_capacity=float(row[5]),
                            battery_level=float(row[6]),
                            # Battery - need to estimate capacity from battery_wh and level
                            battery_capacity=float(row[13]) if len(row) > 13 else (float(row[7]) / float(row[6]) if float(row[6]) > 0 else 10.0),
                            battery_degradation=float(row[14]) if len(row) > 14 else 1.0,
                            # Power consumption - use defaults from parameters
                            idle_power=float(row[15]) if len(row) > 15 else self.parameters.get('Idle_power_min', 0.5),
                            movement_power_factor=float(row[16]) if len(row) > 16 else self.parameters.get('Movement_power_factor_min', 1.5),
                            cpu_power_factor=float(row[17]) if len(row) > 17 else self.parameters.get('CPU_power_factor_min', 2.0),
                            network_power_idle=float(row[18]) if len(row) > 18 else self.parameters.get('Network_power_idle_min', 0.1),
                            network_power_tx=float(row[19]) if len(row) > 19 else self.parameters.get('Network_power_tx_min', 1.0),
                            # Network
                            latency=float(row[8]),
                            reliability=float(row[9]),
                            tx_power=float(row[20]) if len(row) > 20 else self.parameters.get('TX_power_min', 20.0),
                            reliability_window=self.parameters.get('Reliability_window', 50),
                            # State
                            state=bool(row[10].lower() == "true")
                        )
                        # Set battery_wh directly
                        device.battery_wh = float(row[7])
                        
                        # Movement state - read from CSV if available
                        device.is_moving = bool(row[11].lower() == "true") if len(row) > 11 else False
                        device.speed = float(row[12]) if len(row) > 12 else 0.0
                        
                        time_step = int(row[0])
                        if time_step not in recorded_data:
                            recorded_data[time_step] = []
                        recorded_data[time_step].append(device)
            
            self.recorded_data = recorded_data
            self.master_device = MasterDevice(
                x=self.parameters['MasterDeviceX'],
                y=self.parameters['MasterDeviceY'],
                z=self.parameters['MasterDeviceZ']
            )
            
            # Set max time step
            self.max_time_step = max(self.recorded_data.keys()) if self.recorded_data else 0
            
            print(f"Successfully loaded simulation data from {filename}")
            print(f"Time steps available: 0 to {self.max_time_step}")
            print(f"Devices per timestep: {len(recorded_data[0]) if 0 in recorded_data else 0}")
            print(f"Note: Device-specific power parameters set to defaults from environment")
            return recorded_data
            
        except Exception as e:
            print(f"Error loading simulation data: {str(e)}")
            print(f"Row causing error: {row if 'row' in locals() else 'N/A'}")
            raise


    # ===================== REAL-TIME SIMULATION METHODS =====================
    
    def SimMoveFirst(self):
        """
        Initialize Real-Time simulation mode and move to first timestamp
        Must be called after load_simulation_data()
        """
        if not self.recorded_data:
            raise ValueError("No simulation data loaded. Call load_simulation_data() first.")
        
        self.simulation_mode = 'REALTIME'
        self.current_time_step = 0
        
        # Create real-time device instances from first time step
        self.rt_devices = []
        if 0 in self.recorded_data:
            for device in self.recorded_data[0]:
                rt_device = copy(device)
                rt_device.is_selected = False
                self.rt_devices.append(rt_device)
        
        print(f"Real-Time mode activated. Current time step: {self.current_time_step}")
        print(f"Active devices: {len([d for d in self.rt_devices if d.online])}")
        
    def SimMoveNext(self, is_FL_Round=False,Loop=True):
        """
        Move to next timestamp and update all device parameters.
        Applies battery drain for selected devices doing computation.
        Returns True if successful, False if at end.
        if Loop is True return to the first record when reaching the End
        """
        if self.simulation_mode != 'REALTIME':
            raise ValueError("Not in REALTIME mode. Call SimMoveFirst() first.")
        
        self.current_time_step += 1
        if self.current_time_step >= self.max_time_step:
           if Loop:
                 self.current_time_step = 0
           else:
                 return False
        
        if self.current_time_step not in self.recorded_data:
            return False
        
        base_devices = self.recorded_data[self.current_time_step]
        params = self.parameters
        
        # Get parameters with defaults
        comp_power = params.get('Computation_power', 3.0)
        upload_power = params.get('upload_model_power', 3.0)
        upload_time = params.get('upload_model_time', 3)
        download_power = params.get('download_model_power', 2.0)
        download_time = params.get('download_model_time', 2)
        
        min_devices = min(len(base_devices), len(self.rt_devices))
        
        for i in range(min_devices):
            rt_device = self.rt_devices[i]
            base_device = base_devices[i]
            
            was_selected = rt_device.is_selected
            
            # Update attributes
            rt_device.x = base_device.x
            rt_device.y = base_device.y
            rt_device.z = base_device.z
            rt_device.cpu_capacity = base_device.cpu_capacity
            rt_device.latency = base_device.latency
            rt_device.reliability = base_device.reliability
            rt_device.online = base_device.online
            rt_device.speed = base_device.speed
            rt_device.is_moving = base_device.is_moving
            
            # Handle battery updates
            if was_selected and rt_device.online:
                rt_device.update_battery_realistic(1.0, comp_power)
                
                if is_FL_Round:
                    rt_device.update_battery_realistic(upload_time, upload_power)
                
                # Clamp battery
                rt_device.battery_level = max(0.0, min(1.0, rt_device.battery_level))
                rt_device.battery_wh = min(max(0.0, rt_device.battery_wh), rt_device.battery_capacity)
                
                if not rt_device.online:
                    rt_device.is_selected = False
                    print(f"Device {rt_device.device_id} went offline due to battery depletion")
            else:
                rt_device.battery_level = max(0.0, min(1.0, base_device.battery_level))
                rt_device.battery_wh = min(max(0.0, base_device.battery_wh), rt_device.battery_capacity)
            
            # FL round download
            if is_FL_Round and rt_device.online:
                rt_device.update_battery_realistic(download_time, download_power)
                rt_device.battery_level = max(0.0, min(1.0, rt_device.battery_level))
                rt_device.battery_wh = min(max(0.0, rt_device.battery_wh), rt_device.battery_capacity)
                
                if not rt_device.online:
                    rt_device.is_selected = False
                    print(f"Device {rt_device.device_id} went offline due to battery depletion")
            
            rt_device.is_selected = was_selected and rt_device.online
        
        return True
        
    def SimEndRec(self) -> bool:
        """
        Check if we have reached the end of simulation records
        Returns True if at end, False otherwise
        """
        if self.simulation_mode != 'REALTIME':
            raise ValueError("Not in REALTIME mode. Call SimMoveFirst() first.")
        
        return self.current_time_step >= self.max_time_step
    
    def SelectDevice(self, device_id: int) -> bool:
        """
        Select a device for computation (federated learning)
        This will cause additional battery drain in subsequent time steps
        Returns True if successful, False if device not found or offline
        """
        if self.simulation_mode != 'REALTIME':
            raise ValueError("Not in REALTIME mode. Call SimMoveFirst() first.")
        
        for device in self.rt_devices:
            if device.device_id == device_id:
                if device.online:
                    device.is_selected = True
                    print(f"Device {device_id} selected for computation")
                    return True
                else:
                    print(f"Device {device_id} is offline, cannot select")
                    return False
        
        print(f"Device {device_id} not found")
        return False
    
    def UnselectDevice(self, device_id: int) -> bool:
        """
        Unselect a device from computation
        Returns True if successful, False if device not found
        """
        if self.simulation_mode != 'REALTIME':
            raise ValueError("Not in REALTIME mode. Call SimMoveFirst() first.")
        
        for device in self.rt_devices:
            if device.device_id == device_id:
                device.is_selected = False
                print(f"Device {device_id} unselected from computation")
                return True
        
        print(f"Device {device_id} not found")
        return False
    
    def GetRTActiveDevices(self, device_id={}) -> List[Device]:
        """
        Get active devices in Real-Time mode at current timestamp
        Equivalent to GetActiveDevices but uses current_time_step
        """
        if self.simulation_mode != 'REALTIME':
            raise ValueError("Not in REALTIME mode. Call SimMoveFirst() first.")
        
        active_devices = []
        for d in self.rt_devices:
            if d.online:
                if device_id == {}:
                    active_devices.append(d)
                else:
                    if d.device_id in device_id:
                        active_devices.append(d)
        
        return active_devices
    
    def GetRTSelectedDevices(self) -> List[Device]:
        """
        Get devices currently selected for computation
        """
        if self.simulation_mode != 'REALTIME':
            raise ValueError("Not in REALTIME mode. Call SimMoveFirst() first.")
        
        return [d for d in self.rt_devices if d.is_selected and d.online]
    
    def GetCurrentTimeStep(self) -> int:
        """Get current time step in Real-Time mode"""
        return self.current_time_step
    
    def GetSimulationMode(self) -> str:
        """Get current simulation mode"""
        return self.simulation_mode

    def get_devices_at_time(self, time_step: int, online_only: bool = True, 
                          min_cpu: float = None, min_battery: float = None,
                          max_latency: float = None, min_reliability: float = None, 
                          device_ids={}) -> List[Device]:
        """
        Get devices at a specific time step with optional filters
        """
        if time_step not in self.recorded_data:
            return []
            
        filtered_devices = []
        
        for device in self.recorded_data[time_step]:
            if device_ids != {} and device.device_id not in device_ids:
                continue
            if online_only and not device.online:
                continue
            if min_cpu is not None and device.cpu_capacity < min_cpu:
                continue
            if min_battery is not None and device.battery_level < min_battery:
                continue
            if min_reliability is not None and device.reliability < min_reliability:
                continue
            if max_latency is not None and device.latency > max_latency:
                continue
                
            filtered_devices.append(device)
            
        return filtered_devices    
    
    def create_random_devices(self, device_params=None):
        """
        Create random devices with customizable parameters
        Each device gets random values within specified ranges for:
        - Battery capacity and degradation (simulating device age/model)
        - Power consumption characteristics
        - Network transmission power
        - Initial CPU capacity, latency, and reliability
        
        All ranges can be customized via parameters
        """
        if device_params is None:
            device_params = self.parameters
            
        for i in range(1, device_params['Device_count'] + 1):
            # Battery properties (different for each device based on age/model)
            battery_cap = ru(device_params['Battery_capacity_min'], 
                           device_params['Battery_capacity_max'])
            
            battery_level = ru(device_params['Battery_level_min'],
                             device_params['Battery_level_max'])
            
            # Battery degradation: 1.0 = brand new, >1.0 = aged device
            # Older devices drain battery faster due to capacity loss and increased resistance
            battery_deg = ru(device_params['Battery_degradation_min'], 
                           device_params['Battery_degradation_max'])
            
            # Power consumption parameters (vary by device generation/model)
            idle_pwr = ru(device_params['Idle_power_min'], 
                         device_params['Idle_power_max'])
            
            move_pwr = ru(device_params['Movement_power_factor_min'], 
                         device_params['Movement_power_factor_max'])
            
            cpu_pwr = ru(device_params['CPU_power_factor_min'], 
                        device_params['CPU_power_factor_max'])
            
            net_idle = ru(device_params['Network_power_idle_min'], 
                         device_params['Network_power_idle_max'])
            
            net_tx = ru(device_params['Network_power_tx_min'], 
                       device_params['Network_power_tx_max'])
            
            # Network transmission power
            tx_pwr = ru(device_params['TX_power_min'], 
                       device_params['TX_power_max'])
            
            # Initial CPU capacity
            cpu_cap = ru(device_params['CPU_capacity_min'],
                        device_params['CPU_capacity_max'])
            
            # Initial latency (will be recalculated during simulation)
            lat = ru(device_params['Latency_min'],
                    device_params['Latency_max'])
            
            # Initial reliability
            rel = ru(device_params['Reliability_min'],
                    device_params['Reliability_max'])
            
            device = Device(
                device_id=i, 
                x=ru(0, device_params['sWidth']), 
                y=ru(0, device_params['sHeight']), 
                z=ru(0, device_params['sDepth']),
                maxspeed=device_params['MoveSpeed'], 
                acceleration=device_params['Acceleration'], 
                deceleration=device_params['Deceleration'],
                direction_change_prob=device_params['Direction_change_prob'], 
                stop_prob=device_params['Stop_prob'], 
                start_prob=device_params['Start_prob'],
                cpu_capacity=cpu_cap, 
                battery_capacity=battery_cap,
                battery_level=battery_level,
                battery_degradation=battery_deg,
                idle_power=idle_pwr,
                movement_power_factor=move_pwr,
                cpu_power_factor=cpu_pwr,
                network_power_idle=net_idle,
                network_power_tx=net_tx,
                latency=lat, 
                reliability=rel,
                reliability_window=device_params['Reliability_window'],
                tx_power=tx_pwr
            )
            
            # Initialize movement state
            device.is_moving = random.random() < device_params.get('Moving_prob', 0.6)
            self.add_device(device)
        
        print(f"Created {device_params['Device_count']} devices with varying characteristics")


# ===================== EXAMPLE USAGE =====================
if __name__ == "__main__":
    # Example 1: Using default parameters
    print("=" * 80)
    print("EXAMPLE 1: Default Configuration")
    print("=" * 80)
    
    env1 = Environment()
    env1.create_random_devices()
    env1.run_simulation(print_infos=False)
    
    # Plot final state
    env1.PlotSimulationDevices(time_step=99)
    
    # Export data
    env1.export_simulation_data("simulation_default.csv")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 80)
    
    # Example 2: Custom parameters for specific scenario
    custom_params = {
        'MaxTime': 50,
        'Device_count': 10,
        'sWidth': 150,
        'sHeight': 150,
        'sDepth': 75,
        
        # Customize initial device state ranges
        'Battery_level_min': 0.5,  # Start with better charged devices
        'Battery_level_max': 1.0,
        'CPU_capacity_min': 0.2,  # Higher minimum CPU
        'CPU_capacity_max': 0.8,  # Cap maximum CPU
        'Reliability_min': 0.8,  # More reliable devices
        'Reliability_max': 1.0,
        
        # Simulate older devices with faster battery drain
        'Battery_degradation_min': 1.3,
        'Battery_degradation_max': 1.8,
        
        # Higher power consumption devices
        'Idle_power_min': 0.5,
        'Idle_power_max': 1.0,
        'CPU_power_factor_min': 2.0,
        'CPU_power_factor_max': 3.0,
        
        # Urban environment with high path loss
        'Path_loss_exponent': 4.0,
        'Offline_prob': 0.05,
    }
    
    env2 = Environment(params=custom_params)
    env2.create_random_devices()
    env2.run_simulation(print_infos=False)
    
    # Show devices at different time steps
    print("\nDevices at time step 25:")
    env2.PrintSimulationDevices(time_step=25)
    
    # Filter devices with good characteristics
    good_devices = env2.get_devices_at_time(
        time_step=49,
        online_only=True,
        min_battery=0.3,
        min_reliability=0.7,
        max_latency=5.0
    )
    
    print(f"\nFound {len(good_devices)} devices with good characteristics at time 49")
    
    # Create animation
    # env2.create_animation(filename="simulation.gif", fps=5)
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Real-Time Mode for Federated Learning")
    print("=" * 80)
    
    # First, create and save a simulation
    print("\nStep 1: Creating simulation...")
    fl_params = {
        'MaxTime': 30,
        'Device_count': 5,
        'Computation_power': 5.0,  # High power for ML computation (5W)
    }
    
    env3 = Environment(params=fl_params)
    env3.create_random_devices()
    env3.run_simulation(print_infos=False)
    env3.export_simulation_data("fl_simulation.csv")
    
    # Load simulation for real-time use
    print("\nStep 2: Loading simulation in Real-Time mode...")
    env_rt = Environment()
    env_rt.load_simulation_data("fl_simulation.csv")
    
    # Initialize real-time mode
    env_rt.SimMoveFirst()
    
    # Simulate federated learning rounds
    print("\nStep 3: Simulating Federated Learning...")
    print("-" * 80)
    
    round_num = 0
    while not env_rt.SimEndRec():
        round_num += 1
        current_time = env_rt.GetCurrentTimeStep()
        
        print(f"\n=== FL Round {round_num} | Time Step: {current_time} ===")
        
        # Get active devices
        active_devices = env_rt.GetRTActiveDevices()
        print(f"Active devices: {len(active_devices)}")
        
        # Select devices for this round (e.g., those with good battery and reliability)
        selected_count = 0
        for device in active_devices:
            # Selection criteria for federated learning
            if (device.battery_level > 0.4 and 
                device.reliability > 0.7 and 
                device.latency < 10.0 and
                selected_count < 3):  # Limit to 3 devices per round
                
                env_rt.SelectDevice(device.device_id)
                selected_count += 1
                print(f"  ✓ Selected Device {device.device_id}: "
                      f"Battery={device.battery_level:.1%}, "
                      f"Reliability={device.reliability:.1%}, "
                      f"Latency={device.latency:.2f}ms")
        
        # Show selected devices
        selected_devices = env_rt.GetRTSelectedDevices()
        print(f"Total selected for computation: {len(selected_devices)}")
        
        # Move to next time step (this will drain battery for selected devices)
        env_rt.SimMoveNext()
        
        # Show battery levels after computation
        print("\nBattery levels after computation:")
        for device in env_rt.rt_devices:
            status = "SELECTED" if device.is_selected else "idle"
            online_status = "ONLINE" if device.online else "OFFLINE"
            print(f"  Device {device.device_id}: {device.battery_level:.1%} [{status}] [{online_status}]")
        
        # Unselect devices for next round
        for device in selected_devices:
            env_rt.UnselectDevice(device.device_id)
        
        # Stop after 10 rounds for demo
        if round_num >= 10:
            break
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING SIMULATION COMPLETE")
    print(f"Total rounds completed: {round_num}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("ALL SIMULATIONS COMPLETE")
    print("=" * 80)