import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import traci
import os
import csv
import matplotlib.pyplot as plt
from collections import deque
import time
from generate_routes import generate_random_routes

# Constants (matching training script)
PHASES = [0, 1, 2, 3]
STATE_SIZE = 74
CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
                    '-east_0', '-east_0', '-east_1', '-east_2',
                    '-south_0', '-south_0', '-south_1',
                    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
VEHICLES_PER_RUN = 500
SUMO_BINARY = "sumo-gui"  # Use GUI for inference visualization
YELLOW_PHASE_OFFSET = 4
YELLOW_DURATION = 4
MAX_STEPS = 1000

# Duration range for inference (use full trained range)
MIN_DURATION = 5
MAX_DURATION = 30

# --- Policy Network (matching training architecture) ---
class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size=74, hidden_sizes=[128, 256, 128], dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Deeper network with batch normalization
        self.fc1 = tf.keras.layers.Dense(hidden_sizes[0], activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        self.fc2 = tf.keras.layers.Dense(hidden_sizes[1], activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        self.fc3 = tf.keras.layers.Dense(hidden_sizes[2], activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        
        # Separate heads for mean and std of duration distribution
        self.duration_mean = tf.keras.layers.Dense(1, activation='sigmoid')
        self.duration_std = tf.keras.layers.Dense(1, activation='softplus')
        
    def call(self, state, training=False):
        x = self.dropout1(self.bn1(self.fc1(state)), training=training)
        x = self.dropout2(self.bn2(self.fc2(x)), training=training)
        x = self.dropout3(self.bn3(self.fc3(x)), training=training)
        
        mean = self.duration_mean(x)
        std = self.duration_std(x) + 1e-6  # Ensure positive std
        
        return mean, std

# --- Enhanced Helper Functions ---
def get_enhanced_state(current_phase, phase_duration=0):
    """Enhanced state representation with more traffic features"""
    try:
        # Current features - queue lengths
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in CONTROLLED_LANES]
        
        # New features
        vehicle_speeds = []
        waiting_times = []
        approach_vehicles = []
        
        for lane in CONTROLLED_LANES:
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                if veh_ids:
                    speeds = []
                    waits = []
                    approaching = 0
                    
                    for v in veh_ids:
                        try:
                            speed = traci.vehicle.getSpeed(v)
                            wait = traci.vehicle.getWaitingTime(v)
                            speeds.append(speed)
                            waits.append(wait)
                            
                            if speed > 0.1:
                                approaching += 1
                                
                        except traci.exceptions.TraCIException:
                            continue
                    
                    vehicle_speeds.append(np.mean(speeds) if speeds else 0)
                    waiting_times.append(np.mean(waits) if waits else 0)
                    approach_vehicles.append(approaching)
                else:
                    vehicle_speeds.append(0)
                    waiting_times.append(0)
                    approach_vehicles.append(0)
                    
            except traci.exceptions.TraCIException:
                vehicle_speeds.append(0)
                waiting_times.append(0)
                approach_vehicles.append(0)
        
        # Traffic flow rates
        flow_rates = []
        for lane in CONTROLLED_LANES:
            try:
                flow_rates.append(traci.lane.getLastStepVehicleNumber(lane))
            except traci.exceptions.TraCIException:
                flow_rates.append(0)
        
        # Phase timing information
        phase_features = [
            current_phase / 3.0,
            phase_duration / 50.0,
            np.sin(2 * np.pi * current_phase / 4),
            np.cos(2 * np.pi * current_phase / 4)
        ]
        
        # Combine all features
        state = np.array(queue_lengths + vehicle_speeds + waiting_times + 
                        approach_vehicles + flow_rates + phase_features, dtype=np.float32)
        
        # Simple normalization
        state = np.clip(state, -10, 10)
        
        return state
        
    except Exception as e:
        print(f"Error in get_enhanced_state: {e}")
        return np.zeros(STATE_SIZE, dtype=np.float32)

def get_comprehensive_metrics():
    """Get comprehensive traffic metrics for each time step"""
    metrics = {
        'waiting_vehicles': 0,
        'total_queue_length': 0,
        'avg_wait_time': 0,
        'total_vehicles': 0,
        'avg_speed': 0,
        'throughput_this_step': 0
    }
    
    try:
        # Get all vehicles in simulation
        all_vehicles = traci.vehicle.getIDList()
        metrics['total_vehicles'] = len(all_vehicles)
        
        # Get vehicles that arrived this step
        arrived_vehicles = traci.simulation.getArrivedIDList()
        metrics['throughput_this_step'] = len(arrived_vehicles)
        
        if all_vehicles:
            total_wait = 0
            total_speed = 0
            waiting_count = 0
            
            for veh_id in all_vehicles:
                try:
                    wait_time = traci.vehicle.getWaitingTime(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    
                    total_wait += wait_time
                    total_speed += speed
                    
                    if speed < 0.1:  # Vehicle is waiting/stopped
                        waiting_count += 1
                        
                except traci.exceptions.TraCIException:
                    continue
            
            metrics['waiting_vehicles'] = waiting_count
            metrics['avg_wait_time'] = total_wait / len(all_vehicles)
            metrics['avg_speed'] = total_speed / len(all_vehicles)
        
        # Get total queue length across all controlled lanes
        total_queue = 0
        for lane in CONTROLLED_LANES:
            try:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
            except traci.exceptions.TraCIException:
                continue
        
        metrics['total_queue_length'] = total_queue
        
    except Exception as e:
        print(f"Error in get_comprehensive_metrics: {e}")
    
    return metrics

def get_avg_wait_and_queue():
    """Get average waiting time and queue length across all controlled lanes"""
    total_wait = 0
    total_queue = 0
    total_vehicles = 0
    
    for lane in CONTROLLED_LANES:
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            for v in veh_ids:
                try:
                    total_wait += traci.vehicle.getWaitingTime(v)
                    if traci.vehicle.getSpeed(v) < 0.1:
                        total_queue += 1
                    total_vehicles += 1
                except:
                    continue
        except:
            continue
    
    avg_wait = total_wait / total_vehicles if total_vehicles > 0 else 0
    avg_queue = total_queue / len(CONTROLLED_LANES) if CONTROLLED_LANES else 0
    
    return avg_wait, avg_queue

def get_total_vehicles_in_simulation():
    """Get total number of vehicles currently in simulation"""
    try:
        return traci.vehicle.getIDCount()
    except:
        return 0

# --- Inference Class ---
class TrafficLightInference:
    def __init__(self, model_path):
        """Initialize inference with trained model"""
        self.policy_net = PolicyNetwork(state_size=STATE_SIZE)
        
        # Build the model by doing a forward pass first
        dummy_state = tf.zeros((1, STATE_SIZE))
        _ = self.policy_net(dummy_state, training=False)
        print("Model architecture built successfully")
        
        # Load trained weights
        model_loaded = False
        if os.path.exists(model_path):
            try:
                self.policy_net.load_weights(model_path)
                print(f"✓ Loaded trained model from {model_path}")
                model_loaded = True
            except Exception as e:
                print(f"✗ Error loading model weights: {e}")
                print("Trying alternative loading method...")
                
                # Try loading with skip_mismatch
                try:
                    self.policy_net.load_weights(model_path, skip_mismatch=True)
                    print(f"✓ Loaded model with skip_mismatch from {model_path}")
                    model_loaded = True
                except Exception as e2:
                    print(f"✗ Alternative loading also failed: {e2}")
        
        if not model_loaded:
            print(f"⚠ Warning: Could not load model from {model_path}")
            if os.path.exists("logs"):
                print("Available model files in logs/:")
                for f in os.listdir("logs"):
                    if f.endswith('.weights.h5') or f.endswith('.h5'):
                        print(f"  - logs/{f}")
            print("Using randomly initialized model for demonstration")
        
        self.model_loaded = model_loaded
        
    def predict_duration(self, state, use_mean=False):
        """Predict phase duration from state"""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mean, std = self.policy_net(state_tensor, training=False)
        
        if use_mean:
            # Use mean for deterministic behavior
            normalized_duration = mean[0][0]
        else:
            # Sample from distribution
            dist = tfp.distributions.Normal(mean[0][0], std[0][0])
            normalized_duration = dist.sample()
        
        # Clip and scale to duration range
        normalized_duration = tf.clip_by_value(normalized_duration, 0.0, 1.0)
        duration = int(normalized_duration * (MAX_DURATION - MIN_DURATION) + MIN_DURATION)
        
        return duration, float(mean[0][0]), float(std[0][0])
    
    def run_inference(self, num_runs=3, use_mean=False, save_results=True):
        """Run inference for multiple simulation runs"""
        model_status = "trained" if self.model_loaded else "randomly initialized"
        print(f"Starting inference with {model_status} model...")
        print(f"Number of runs: {num_runs}")
        print(f"Use mean predictions: {use_mean}")
        
        all_run_metrics = []
        
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")
            
            # Generate routes for this run
            route_file = "../code for rl/random_routes.rou.xml"
            generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)
            
            run_metrics = self._single_run(run, use_mean)
            all_run_metrics.append(run_metrics)
            
            print(f"Run {run + 1} completed:")
            print(f"  Total throughput: {run_metrics['total_throughput']}")
            print(f"  Average wait time: {run_metrics['avg_wait_time']:.2f}s")
            print(f"  Average queue length: {run_metrics['avg_queue_length']:.2f}")
            print(f"  Phases completed: {run_metrics['phases_completed']}")
            print(f"  Average phase duration: {run_metrics['avg_phase_duration']:.1f}s")
            print(f"  Average speed: {run_metrics['avg_speed']:.2f}m/s")
        
        # Calculate summary statistics
        self._print_summary_statistics(all_run_metrics)
        
        if save_results:
            self._save_results(all_run_metrics)
            
        return all_run_metrics
    
    def _single_run(self, run_id, use_mean):
        """Run a single inference simulation"""
        try:
            traci.start([SUMO_BINARY, "-c", "../trafficinter.sumocfg"])
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            return None
        
        current_phase = 0
        step = 0
        total_throughput = 0
        cumulative_throughput = []
        
        # Time series data for plots
        time_series_data = {
            'waiting_vehicles': [],
            'queue_lengths': [],
            'wait_times': [],
            'total_vehicles': [],
            'throughput_cumulative': [],
            'avg_speeds': [],
            'step_numbers': []
        }
        
        # Phase-based data
        phase_data = {
            'durations': [],
            'phase_numbers': []
        }
        
        phase_counter = 0
        
        try:
            while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
                # Get current state
                state = get_enhanced_state(current_phase, 0)
                
                # Predict duration
                duration, mean_pred, std_pred = self.predict_duration(state, use_mean)
                phase_data['durations'].append(duration)
                phase_data['phase_numbers'].append(phase_counter)
                
                print(f"Phase {current_phase}: Duration={duration}s (mean={mean_pred:.3f}, std={std_pred:.3f})")
                
                # Apply green phase
                traci.trafficlight.setPhase(TLS_ID, current_phase)
                vehicles_passed_this_phase = 0
                
                for phase_step in range(duration):
                    if step >= MAX_STEPS:
                        break
                    
                    # Get comprehensive metrics for this step
                    metrics = get_comprehensive_metrics()
                    
                    # Count vehicles that complete their journey
                    arrived_this_step = traci.simulation.getArrivedIDList()
                    vehicles_passed_this_step = len(arrived_this_step)
                    total_throughput += vehicles_passed_this_step
                    
                    # Store time series data
                    time_series_data['waiting_vehicles'].append(metrics['waiting_vehicles'])
                    time_series_data['queue_lengths'].append(metrics['total_queue_length'])
                    time_series_data['wait_times'].append(metrics['avg_wait_time'])
                    time_series_data['total_vehicles'].append(metrics['total_vehicles'])
                    time_series_data['throughput_cumulative'].append(total_throughput)
                    time_series_data['avg_speeds'].append(metrics['avg_speed'])
                    time_series_data['step_numbers'].append(step)
                    
                    traci.simulationStep()
                    step += 1
                
                if step >= MAX_STEPS:
                    break
                
                # Apply yellow phase
                yellow_phase = current_phase + YELLOW_PHASE_OFFSET
                traci.trafficlight.setPhase(TLS_ID, yellow_phase)
                for _ in range(YELLOW_DURATION):
                    if step >= MAX_STEPS:
                        break
                    
                    # Get comprehensive metrics for this step
                    metrics = get_comprehensive_metrics()
                    
                    arrived_this_step = traci.simulation.getArrivedIDList()
                    vehicles_passed_this_step = len(arrived_this_step)
                    total_throughput += vehicles_passed_this_step
                    
                    # Store time series data
                    time_series_data['waiting_vehicles'].append(metrics['waiting_vehicles'])
                    time_series_data['queue_lengths'].append(metrics['total_queue_length'])
                    time_series_data['wait_times'].append(metrics['avg_wait_time'])
                    time_series_data['total_vehicles'].append(metrics['total_vehicles'])
                    time_series_data['throughput_cumulative'].append(total_throughput)
                    time_series_data['avg_speeds'].append(metrics['avg_speed'])
                    time_series_data['step_numbers'].append(step)
                    
                    traci.simulationStep()
                    step += 1
                
                if step >= MAX_STEPS:
                    break
                
                current_phase = (current_phase + 1) % 4
                phase_counter += 1
                
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            try:
                traci.close()
            except:
                pass
        
        # Calculate run metrics
        run_metrics = {
            'run_id': run_id,
            'total_throughput': total_throughput,
            'avg_wait_time': np.mean(time_series_data['wait_times']) if time_series_data['wait_times'] else 0,
            'avg_queue_length': np.mean(time_series_data['queue_lengths']) if time_series_data['queue_lengths'] else 0,
            'avg_speed': np.mean(time_series_data['avg_speeds']) if time_series_data['avg_speeds'] else 0,
            'phases_completed': len(phase_data['durations']),
            'avg_phase_duration': np.mean(phase_data['durations']) if phase_data['durations'] else 0,
            'max_wait_time': np.max(time_series_data['wait_times']) if time_series_data['wait_times'] else 0,
            'max_queue_length': np.max(time_series_data['queue_lengths']) if time_series_data['queue_lengths'] else 0,
            'steps_completed': step,
            'time_series_data': time_series_data,
            'phase_data': phase_data
        }
        
        return run_metrics
    
    def _print_summary_statistics(self, all_run_metrics):
        """Print summary statistics across all runs"""
        print("\n" + "="*50)
        print("INFERENCE SUMMARY STATISTICS")
        print("="*50)
        
        throughputs = [m['total_throughput'] for m in all_run_metrics]
        wait_times = [m['avg_wait_time'] for m in all_run_metrics]
        queue_lengths = [m['avg_queue_length'] for m in all_run_metrics]
        phase_durations = [m['avg_phase_duration'] for m in all_run_metrics]
        avg_speeds = [m['avg_speed'] for m in all_run_metrics]
        
        print(f"Throughput - Mean: {np.mean(throughputs):.1f} ± {np.std(throughputs):.1f}")
        print(f"Wait Time - Mean: {np.mean(wait_times):.2f}s ± {np.std(wait_times):.2f}s")
        print(f"Queue Length - Mean: {np.mean(queue_lengths):.2f} ± {np.std(queue_lengths):.2f}")
        print(f"Average Speed - Mean: {np.mean(avg_speeds):.2f}m/s ± {np.std(avg_speeds):.2f}m/s")
        print(f"Phase Duration - Mean: {np.mean(phase_durations):.1f}s ± {np.std(phase_durations):.1f}s")
        print("="*50)
    
    def _save_results(self, all_run_metrics):
        """Save results to CSV and create plots"""
        os.makedirs("inference_results", exist_ok=True)
        
        # Save summary statistics
        with open("inference_results/summary_statistics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Run", "TotalThroughput", "AvgWaitTime", "AvgQueueLength", 
                           "AvgSpeed", "PhasesCompleted", "AvgPhaseDuration", "MaxWaitTime", "MaxQueueLength"])
            
            for metrics in all_run_metrics:
                writer.writerow([
                    metrics['run_id'] + 1,
                    metrics['total_throughput'],
                    metrics['avg_wait_time'],
                    metrics['avg_queue_length'],
                    metrics['avg_speed'],
                    metrics['phases_completed'],
                    metrics['avg_phase_duration'],
                    metrics['max_wait_time'],
                    metrics['max_queue_length']
                ])
        
        # Save detailed time series data
        self._save_time_series_data(all_run_metrics)
        
        # Create plots
        self._create_comprehensive_plots(all_run_metrics)
        
        print(f"\nResults saved to 'inference_results/' directory")
    
    def _save_time_series_data(self, all_run_metrics):
        """Save detailed time series data for each run"""
        for i, metrics in enumerate(all_run_metrics):
            filename = f"inference_results/time_series_run_{i+1}.csv"
            time_data = metrics['time_series_data']
            
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "WaitingVehicles", "QueueLength", "AvgWaitTime", 
                               "TotalVehicles", "CumulativeThroughput", "AvgSpeed"])
                
                for j in range(len(time_data['step_numbers'])):
                    writer.writerow([
                        time_data['step_numbers'][j],
                        time_data['waiting_vehicles'][j],
                        time_data['queue_lengths'][j],
                        time_data['wait_times'][j],
                        time_data['total_vehicles'][j],
                        time_data['throughput_cumulative'][j],
                        time_data['avg_speeds'][j]
                    ])
    
    def _create_comprehensive_plots(self, all_run_metrics):
        """Create comprehensive visualization plots"""
        # Set up the plotting style
        plt.style.use('default')
        
        # Create time series plots (3x3 grid)
        fig1, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig1.suptitle('Traffic Light Control - Time Series Analysis', fontsize=16)
        
        # Use first run for time series plots
        if all_run_metrics:
            time_data = all_run_metrics[0]['time_series_data']
            steps = time_data['step_numbers']
            
            # Plot 1: Waiting vehicles vs time step
            axes[0,0].plot(steps, time_data['waiting_vehicles'], 'b-', linewidth=1)
            axes[0,0].set_title('Waiting Vehicles vs Time Step')
            axes[0,0].set_xlabel('Time Step')
            axes[0,0].set_ylabel('Number of Waiting Vehicles')
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Queue length vs time step
            axes[0,1].plot(steps, time_data['queue_lengths'], 'r-', linewidth=1)
            axes[0,1].set_title('Queue Length vs Time Step')
            axes[0,1].set_xlabel('Time Step')
            axes[0,1].set_ylabel('Total Queue Length')
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Wait times vs time step
            axes[0,2].plot(steps, time_data['wait_times'], 'g-', linewidth=1)
            axes[0,2].set_title('Average Wait Time vs Time Step')
            axes[0,2].set_xlabel('Time Step')
            axes[0,2].set_ylabel('Average Wait Time (s)')
            axes[0,2].grid(True, alpha=0.3)
            
            # Plot 4: Total vehicles vs time step
            axes[1,0].plot(steps, time_data['total_vehicles'], 'm-', linewidth=1)
            axes[1,0].set_title('Total Vehicles vs Time Step')
            axes[1,0].set_xlabel('Time Step')
            axes[1,0].set_ylabel('Number of Vehicles in Simulation')
            axes[1,0].grid(True, alpha=0.3)
            
            # Plot 5: Cumulative throughput vs time step
            axes[1,1].plot(steps, time_data['throughput_cumulative'], 'c-', linewidth=1)
            axes[1,1].set_title('Cumulative Throughput vs Time Step')
            axes[1,1].set_xlabel('Time Step')
            axes[1,1].set_ylabel('Cumulative Vehicles Processed')
            axes[1,1].grid(True, alpha=0.3)
            
            # Plot 6: Average speed vs time step
            axes[1,2].plot(steps, time_data['avg_speeds'], 'orange', linewidth=1)
            axes[1,2].set_title('Average Speed vs Time Step')
            axes[1,2].set_xlabel('Time Step')
            axes[1,2].set_ylabel('Average Speed (m/s)')
            axes[1,2].grid(True, alpha=0.3)
            
            # Plot 7: Phase duration vs phase number
            phase_data = all_run_metrics[0]['phase_data']
            axes[2,0].plot(phase_data['phase_numbers'], phase_data['durations'], 'ko-', markersize=4)
            axes[2,0].set_title('Phase Duration vs Phase Number')
            axes[2,0].set_xlabel('Phase Number')
            axes[2,0].set_ylabel('Duration (s)')
            axes[2,0].grid(True, alpha=0.3)
            
            # Plot 8: Average waiting time vs run (across all runs)
            run_wait_times = [m['avg_wait_time'] for m in all_run_metrics]
            run_numbers = list(range(1, len(run_wait_times) + 1))
            axes[2,1].bar(run_numbers, run_wait_times, color='lightblue', edgecolor='blue')
            axes[2,1].set_title('Average Waiting Time vs Run')
            axes[2,1].set_xlabel('Run Number')
            axes[2,1].set_ylabel('Average Wait Time (s)')
            axes[2,1].grid(True, alpha=0.3)
            
            # Plot 9: Throughput vs run (across all runs)
            run_throughputs = [m['total_throughput'] for m in all_run_metrics]
            axes[2,2].bar(run_numbers, run_throughputs, color='lightgreen', edgecolor='green')
            axes[2,2].set_title('Total Throughput vs Run')
            axes[2,2].set_xlabel('Run Number')
            axes[2,2].set_ylabel('Total Vehicles Processed')
            axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('inference_results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional comparison plots
        self._create_comparison_plots(all_run_metrics)
    
    def _create_comparison_plots(self, all_run_metrics):
        """Create comparison plots across all runs"""
        if len(all_run_metrics) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Multi-Run Comparison Analysis', fontsize=14)
        
        # Plot 1: Queue length comparison across runs
        for i, metrics in enumerate(all_run_metrics):
            time_data = metrics['time_series_data']
            axes[0,0].plot(time_data['step_numbers'], time_data['queue_lengths'], 
                          label=f'Run {i+1}', linewidth=1)
        axes[0,0].set_title('Queue Length Comparison')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Queue Length')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Average speed comparison across runs
        for i, metrics in enumerate(all_run_metrics):
            time_data = metrics['time_series_data']
            axes[0,1].plot(time_data['step_numbers'], time_data['avg_speeds'], 
                          label=f'Run {i+1}', linewidth=1)
        axes[0,1].set_title('Average Speed Comparison')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Average Speed (m/s)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative throughput comparison
        for i, metrics in enumerate(all_run_metrics):
            time_data = metrics['time_series_data']
            axes[1,0].plot(time_data['step_numbers'], time_data['throughput_cumulative'], 
                          label=f'Run {i+1}', linewidth=1)
        axes[1,0].set_title('Cumulative Throughput Comparison')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('Cumulative Throughput')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics summary
        metrics_names = ['Throughput', 'Avg Wait Time', 'Avg Queue Length', 'Avg Speed']
        run_numbers = list(range(1, len(all_run_metrics) + 1))
        
        # Normalize metrics for comparison (0-1 scale)
        throughputs = [m['total_throughput'] for m in all_run_metrics]
        wait_times = [m['avg_wait_time'] for m in all_run_metrics]
        queue_lengths = [m['avg_queue_length'] for m in all_run_metrics]
        avg_speeds = [m['avg_speed'] for m in all_run_metrics]
        
        # Normalize to 0-1 scale for comparison
        def normalize(data):
            if max(data) == min(data):
                return [0.5] * len(data)
            return [(x - min(data)) / (max(data) - min(data)) for x in data]
        
        norm_throughput = normalize(throughputs)
        norm_wait = [1 - x for x in normalize(wait_times)]  # Invert (lower is better)
        norm_queue = [1 - x for x in normalize(queue_lengths)]  # Invert (lower is better)
        norm_speed = normalize(avg_speeds)
        
        x = np.arange(len(run_numbers))
        width = 0.2
        
        axes[1,1].bar(x - 1.5*width, norm_throughput, width, label='Throughput (normalized)', alpha=0.8)
        axes[1,1].bar(x - 0.5*width, norm_wait, width, label='Wait Time (inverted)', alpha=0.8)
        axes[1,1].bar(x + 0.5*width, norm_queue, width, label='Queue Length (inverted)', alpha=0.8)
        axes[1,1].bar(x + 1.5*width, norm_speed, width, label='Avg Speed (normalized)', alpha=0.8)
        
        axes[1,1].set_title('Normalized Performance Metrics')
        axes[1,1].set_xlabel('Run Number')
        axes[1,1].set_ylabel('Normalized Score (0-1, higher is better)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(run_numbers)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('inference_results/multi_run_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

# --- Main Function ---
def main():
    """Main inference function"""
    # Configuration
    MODEL_PATHS = [
        "logs/policy_model_final.weights.h5"
    ]
    
    NUM_RUNS = 3  # Number of simulation runs
    USE_MEAN = False  # Whether to use mean predictions (deterministic) or sample from distribution
    
    print("Traffic Light Control - Enhanced Inference Mode")
    print("==============================================")
    
    # Find the best available model
    MODEL_PATH = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            MODEL_PATH = path
            break
    
    if MODEL_PATH is None:
        print("No trained model found. Checking logs directory...")
        if os.path.exists("logs"):
            model_files = [f for f in os.listdir("logs") if f.endswith('.weights.h5')]
            if model_files:
                MODEL_PATH = os.path.join("logs", model_files[0])
                print(f"Using first available model: {MODEL_PATH}")
            else:
                print("No .weights.h5 files found in logs/")
                MODEL_PATH = "dummy_path"  # Will trigger random initialization
        else:
            print("logs/ directory not found")
            MODEL_PATH = "dummy_path"  # Will trigger random initialization
    
    print(f"Model path: {MODEL_PATH}")
    
    # Initialize inference
    inference = TrafficLightInference(MODEL_PATH)
    
    # Run inference
    results = inference.run_inference(
        num_runs=NUM_RUNS,
        use_mean=USE_MEAN,
        save_results=True
    )
    
    print("\nInference completed successfully!")
    print("Generated plots:")
    print("  - comprehensive_analysis.png: 9 time series and summary plots")
    print("  - multi_run_comparison.png: Comparison across multiple runs")
    print("Check 'inference_results/' directory for:")
    print("  - Detailed CSV files with time series data")
    print("  - Summary statistics")
    print("  - Visualization plots")
    
    # Print final performance summary
    if results:
        print(f"\nFINAL PERFORMANCE SUMMARY:")
        print(f"Average across {len(results)} runs:")
        avg_throughput = np.mean([r['total_throughput'] for r in results])
        avg_wait = np.mean([r['avg_wait_time'] for r in results])
        avg_speed = np.mean([r['avg_speed'] for r in results])
        avg_queue = np.mean([r['avg_queue_length'] for r in results])
        
        print(f"  Throughput: {avg_throughput:.1f} vehicles")
        print(f"  Wait Time: {avg_wait:.2f} seconds")
        print(f"  Average Speed: {avg_speed:.2f} m/s")
        print(f"  Queue Length: {avg_queue:.2f} vehicles")

if __name__ == "__main__":
    main()