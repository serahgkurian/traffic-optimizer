import numpy as np
import traci
import os
import csv
import matplotlib.pyplot as plt
from collections import deque
import time
from generate_routes import generate_random_routes

# Constants (matching training script and TLS configuration)
# Updated to match the 8-phase TLS configuration
PHASES = [0, 1, 2, 3, 4, 5, 6, 7]  # All 8 phases from TLS file
GREEN_PHASES = [0, 2, 4, 6]  # Green phases only
YELLOW_PHASES = [1, 3, 5, 7]  # Yellow phases only
STATE_SIZE = 74
CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
                    '-east_0', '-east_0', '-east_1', '-east_2',
                    '-south_0', '-south_0', '-south_1',
                    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
VEHICLES_PER_RUN = 500
SUMO_BINARY = "sumo-gui"  # Use GUI for visualization
YELLOW_DURATION = 4  # Fixed yellow duration from TLS file
MAX_STEPS = 1000

# Fixed timing parameters for baseline
FIXED_GREEN_DURATION = 22  # Match TLS file default duration
ADAPTIVE_MODE = False  # Set to True for simple adaptive control

# Phase mapping based on TLS configuration
PHASE_MAPPING = {
    0: "East-West Straight and Right",  # Phase 0: East-West Straight and right
    1: "East-West Yellow",              # Phase 1: East-West Yellow
    2: "East-West Left",                # Phase 2: East-West Left
    3: "East-West Left Yellow",         # Phase 3: East-West Left Yellow
    4: "North-South Straight and Right", # Phase 4: North-South Straight and right
    5: "North-South Yellow",            # Phase 5: North-South Yellow
    6: "North-South Left",              # Phase 6: North-South Left
    7: "North-South Left Yellow"        # Phase 7: North-South Left Yellow
}

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
        
        # Phase timing information - updated for 8 phases
        phase_features = [
            current_phase / 7.0,  # Normalize by max phase (7)
            phase_duration / 50.0,
            np.sin(2 * np.pi * current_phase / 8),  # 8-phase cycle
            np.cos(2 * np.pi * current_phase / 8)
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

def get_phase_queue_length(phase):
    """Get queue length for lanes corresponding to a specific phase"""
    # Updated phase-to-lane mapping based on TLS configuration
    phase_lanes = {
        0: ['-east_0', '-east_1', '-east_2', '-west_0', '-west_1', '-west_2'],    # East-West Straight and Right
        1: ['-east_0', '-east_1', '-east_2', '-west_0', '-west_1', '-west_2'],    # East-West Yellow
        2: ['-east_0', '-west_0'],                                                 # East-West Left
        3: ['-east_0', '-west_0'],                                                 # East-West Left Yellow  
        4: ['-north_0', '-north_1', '-south_0', '-south_1'],                      # North-South Straight and Right
        5: ['-north_0', '-north_1', '-south_0', '-south_1'],                      # North-South Yellow
        6: ['-north_0', '-south_0'],                                               # North-South Left
        7: ['-north_0', '-south_0']                                                # North-South Left Yellow
    }
    
    total_queue = 0
    if phase in phase_lanes:
        for lane in phase_lanes[phase]:
            try:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
            except traci.exceptions.TraCIException:
                continue
    
    return total_queue

def get_adaptive_duration(current_phase):
    """Simple adaptive duration based on queue length - only for green phases"""
    if not ADAPTIVE_MODE or current_phase not in GREEN_PHASES:
        # Return fixed duration for yellow phases or when not in adaptive mode
        if current_phase in YELLOW_PHASES:
            return YELLOW_DURATION
        else:
            return FIXED_GREEN_DURATION
    
    # Get queue length for current green phase
    queue_length = get_phase_queue_length(current_phase)
    
    # Adaptive logic: extend green time if there's a queue
    base_duration = FIXED_GREEN_DURATION
    if queue_length > 5:
        # Extend green time for longer queues
        extension = min(queue_length * 2, 15)  # Max 15 seconds extension
        return base_duration + extension
    elif queue_length == 0:
        # Reduce green time if no queue
        return max(base_duration - 5, 10)  # Minimum 10 seconds
    else:
        return base_duration

# --- Baseline Traffic Light Controller ---
class BaselineTrafficLight:
    def __init__(self, control_type="fixed"):
        """
        Initialize baseline traffic light controller
        control_type: "fixed" for fixed timing, "adaptive" for simple adaptive control
        """
        self.control_type = control_type
        global ADAPTIVE_MODE
        ADAPTIVE_MODE = (control_type == "adaptive")
        
        print(f"Initialized {control_type} traffic light controller")
        print(f"Phase configuration:")
        for phase, description in PHASE_MAPPING.items():
            print(f"  Phase {phase}: {description}")
        
        if control_type == "fixed":
            print(f"Fixed green duration: {FIXED_GREEN_DURATION} seconds")
            print(f"Fixed yellow duration: {YELLOW_DURATION} seconds")
        else:
            print(f"Adaptive control with base duration: {FIXED_GREEN_DURATION} seconds")
        
    def get_phase_duration(self, current_phase):
        """Get phase duration based on control type and phase"""
        if self.control_type == "fixed":
            if current_phase in YELLOW_PHASES:
                return YELLOW_DURATION
            else:
                return FIXED_GREEN_DURATION
        else:
            return get_adaptive_duration(current_phase)
    
    def run_baseline(self, num_runs=3, save_results=True):
        """Run baseline traffic light control for multiple simulation runs"""
        print(f"Starting baseline traffic light control ({self.control_type})...")
        print(f"Number of runs: {num_runs}")
        
        all_run_metrics = []
        
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")
            
            # Generate routes for this run
            route_file = "../code for rl/random_routes.rou.xml"
            generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)
            
            run_metrics = self._single_run(run)
            if run_metrics:
                all_run_metrics.append(run_metrics)
                
                print(f"Run {run + 1} completed:")
                print(f"  Total throughput: {run_metrics['total_throughput']}")
                print(f"  Average wait time: {run_metrics['avg_wait_time']:.2f}s")
                print(f"  Average queue length: {run_metrics['avg_queue_length']:.2f}")
                print(f"  Full cycles completed: {run_metrics['full_cycles_completed']}")
                print(f"  Average green phase duration: {run_metrics['avg_green_phase_duration']:.1f}s")
                print(f"  Average speed: {run_metrics['avg_speed']:.2f}m/s")
            else:
                print(f"Run {run + 1} failed")
        
        if all_run_metrics:
            # Calculate summary statistics
            self._print_summary_statistics(all_run_metrics)
            
            if save_results:
                self._save_results(all_run_metrics)
                
        return all_run_metrics
    
    def _single_run(self, run_id):
        """Run a single baseline simulation"""
        try:
            traci.start([SUMO_BINARY, "-c", "../trafficinter.sumocfg"])
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            return None
        
        current_phase = 0  # Start with phase 0 (East-West Straight and Right)
        step = 0
        total_throughput = 0
        
        # Time series data for plots
        time_series_data = {
            'waiting_vehicles': [],
            'queue_lengths': [],
            'wait_times': [],
            'total_vehicles': [],
            'throughput_cumulative': [],
            'avg_speeds': [],
            'step_numbers': [],
            'current_phases': []
        }
        
        # Phase-based data
        phase_data = {
            'durations': [],
            'phase_numbers': [],
            'phase_types': []  # Green or Yellow
        }
        
        phase_counter = 0
        full_cycles = 0
        
        try:
            while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
                # Get phase duration
                duration = self.get_phase_duration(current_phase)
                phase_data['durations'].append(duration)
                phase_data['phase_numbers'].append(current_phase)
                phase_data['phase_types'].append('Green' if current_phase in GREEN_PHASES else 'Yellow')
                
                phase_description = PHASE_MAPPING[current_phase]
                
                if self.control_type == "adaptive" and current_phase in GREEN_PHASES:
                    queue_length = get_phase_queue_length(current_phase)
                    print(f"Phase {current_phase} ({phase_description}): Duration={duration}s (queue={queue_length})")
                else:
                    print(f"Phase {current_phase} ({phase_description}): Duration={duration}s")
                
                # Apply the phase
                traci.trafficlight.setPhase(TLS_ID, current_phase)
                
                # Run this phase for the calculated duration
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
                    time_series_data['current_phases'].append(current_phase)
                    
                    traci.simulationStep()
                    step += 1
                
                if step >= MAX_STEPS:
                    break
                
                # Move to next phase in sequence (0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0)
                current_phase = (current_phase + 1) % 8
                phase_counter += 1
                
                # Count full cycles (complete 8-phase cycles)
                if current_phase == 0 and phase_counter > 0:
                    full_cycles += 1
                
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            try:
                traci.close()
            except:
                pass
        
        # Calculate run metrics
        green_durations = [d for i, d in enumerate(phase_data['durations']) 
                          if phase_data['phase_types'][i] == 'Green']
        
        run_metrics = {
            'run_id': run_id,
            'control_type': self.control_type,
            'total_throughput': total_throughput,
            'avg_wait_time': np.mean(time_series_data['wait_times']) if time_series_data['wait_times'] else 0,
            'avg_queue_length': np.mean(time_series_data['queue_lengths']) if time_series_data['queue_lengths'] else 0,
            'avg_speed': np.mean(time_series_data['avg_speeds']) if time_series_data['avg_speeds'] else 0,
            'total_phases_completed': len(phase_data['durations']),
            'full_cycles_completed': full_cycles,
            'avg_green_phase_duration': np.mean(green_durations) if green_durations else 0,
            'avg_yellow_phase_duration': YELLOW_DURATION,
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
        print(f"BASELINE SUMMARY STATISTICS ({self.control_type.upper()})")
        print("="*50)
        
        throughputs = [m['total_throughput'] for m in all_run_metrics]
        wait_times = [m['avg_wait_time'] for m in all_run_metrics]
        queue_lengths = [m['avg_queue_length'] for m in all_run_metrics]
        green_phase_durations = [m['avg_green_phase_duration'] for m in all_run_metrics]
        avg_speeds = [m['avg_speed'] for m in all_run_metrics]
        full_cycles = [m['full_cycles_completed'] for m in all_run_metrics]
        
        print(f"Throughput - Mean: {np.mean(throughputs):.1f} ± {np.std(throughputs):.1f}")
        print(f"Wait Time - Mean: {np.mean(wait_times):.2f}s ± {np.std(wait_times):.2f}s")
        print(f"Queue Length - Mean: {np.mean(queue_lengths):.2f} ± {np.std(queue_lengths):.2f}")
        print(f"Average Speed - Mean: {np.mean(avg_speeds):.2f}m/s ± {np.std(avg_speeds):.2f}m/s")
        print(f"Green Phase Duration - Mean: {np.mean(green_phase_durations):.1f}s ± {np.std(green_phase_durations):.1f}s")
        print(f"Full Cycles Completed - Mean: {np.mean(full_cycles):.1f} ± {np.std(full_cycles):.1f}")
        print("="*50)
    
    def _save_results(self, all_run_metrics):
        """Save results to CSV and create plots"""
        results_dir = f"baseline_results_{self.control_type}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save summary statistics
        with open(f"{results_dir}/summary_statistics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Run", "ControlType", "TotalThroughput", "AvgWaitTime", "AvgQueueLength", 
                           "AvgSpeed", "TotalPhasesCompleted", "FullCyclesCompleted", "AvgGreenPhaseDuration", 
                           "MaxWaitTime", "MaxQueueLength"])
            
            for metrics in all_run_metrics:
                writer.writerow([
                    metrics['run_id'] + 1,
                    metrics['control_type'],
                    metrics['total_throughput'],
                    metrics['avg_wait_time'],
                    metrics['avg_queue_length'],
                    metrics['avg_speed'],
                    metrics['total_phases_completed'],
                    metrics['full_cycles_completed'],
                    metrics['avg_green_phase_duration'],
                    metrics['max_wait_time'],
                    metrics['max_queue_length']
                ])
        
        # Save detailed time series data
        self._save_time_series_data(all_run_metrics, results_dir)
        
        # Create plots
        self._create_comprehensive_plots(all_run_metrics, results_dir)
        
        print(f"\nResults saved to '{results_dir}/' directory")
    
    def _save_time_series_data(self, all_run_metrics, results_dir):
        """Save detailed time series data for each run"""
        for i, metrics in enumerate(all_run_metrics):
            filename = f"{results_dir}/time_series_run_{i+1}.csv"
            time_data = metrics['time_series_data']
            
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "WaitingVehicles", "QueueLength", "AvgWaitTime", 
                               "TotalVehicles", "CumulativeThroughput", "AvgSpeed", "CurrentPhase"])
                
                for j in range(len(time_data['step_numbers'])):
                    writer.writerow([
                        time_data['step_numbers'][j],
                        time_data['waiting_vehicles'][j],
                        time_data['queue_lengths'][j],
                        time_data['wait_times'][j],
                        time_data['total_vehicles'][j],
                        time_data['throughput_cumulative'][j],
                        time_data['avg_speeds'][j],
                        time_data['current_phases'][j]
                    ])
    
    def _create_comprehensive_plots(self, all_run_metrics, results_dir):
        """Create comprehensive visualization plots"""
        # Set up the plotting style
        plt.style.use('default')
        
        # Create time series plots (3x3 grid)
        fig1, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig1.suptitle(f'Baseline Traffic Light Control ({self.control_type.title()}) - Time Series Analysis', fontsize=16)
        
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
            
            # Plot 7: Phase progression vs time step
            axes[2,0].plot(steps, time_data['current_phases'], 'ko-', markersize=2)
            axes[2,0].set_title('Phase Progression vs Time Step')
            axes[2,0].set_xlabel('Time Step')
            axes[2,0].set_ylabel('Current Phase')
            axes[2,0].set_yticks(range(8))
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
            run_numbers = list(range(1, len(run_throughputs) + 1))
            axes[2,2].bar(run_numbers, run_throughputs, color='lightgreen', edgecolor='green')
            axes[2,2].set_title('Total Throughput vs Run')
            axes[2,2].set_xlabel('Run Number')
            axes[2,2].set_ylabel('Total Vehicles Processed')
            axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional comparison plots
        if len(all_run_metrics) > 1:
            self._create_comparison_plots(all_run_metrics, results_dir)
    
    def _create_comparison_plots(self, all_run_metrics, results_dir):
        """Create comparison plots across all runs"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Multi-Run Comparison Analysis ({self.control_type.title()})', fontsize=14)
        
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
        run_numbers = list(range(1, len(all_run_metrics) + 1))
        throughputs = [m['total_throughput'] for m in all_run_metrics]
        wait_times = [m['avg_wait_time'] for m in all_run_metrics]
        
        # Create a dual-axis plot
        ax1 = axes[1,1]
        ax2 = ax1.twinx()
        
        bars1 = ax1.bar([x - 0.2 for x in run_numbers], throughputs, 0.4, 
                       label='Throughput', color='lightblue', alpha=0.7)
        bars2 = ax2.bar([x + 0.2 for x in run_numbers], wait_times, 0.4, 
                       label='Avg Wait Time', color='lightcoral', alpha=0.7)
        
        ax1.set_xlabel('Run Number')
        ax1.set_ylabel('Total Throughput', color='blue')
        ax2.set_ylabel('Avg Wait Time (s)', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.set_title('Performance Metrics Summary')
        ax1.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/multi_run_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

# --- Main Execution Functions ---
def compare_baselines():
    """Compare fixed and adaptive baseline controllers"""
    print("Comparing Fixed vs Adaptive Baseline Controllers")
    print("=" * 60)
    
    # Run fixed timing baseline
    print("\n1. Running Fixed Timing Baseline...")
    fixed_controller = BaselineTrafficLight(control_type="fixed")
    fixed_results = fixed_controller.run_baseline(num_runs=3, save_results=True)
    
    # Run adaptive timing baseline  
    print("\n2. Running Adaptive Timing Baseline...")
    adaptive_controller = BaselineTrafficLight(control_type="adaptive")
    adaptive_results = adaptive_controller.run_baseline(num_runs=3, save_results=True)
    
    # Compare results
    if fixed_results and adaptive_results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        # Fixed results summary
        fixed_throughput = np.mean([m['total_throughput'] for m in fixed_results])
        fixed_wait_time = np.mean([m['avg_wait_time'] for m in fixed_results])
        fixed_queue_length = np.mean([m['avg_queue_length'] for m in fixed_results])
        fixed_speed = np.mean([m['avg_speed'] for m in fixed_results])
        
        # Adaptive results summary
        adaptive_throughput = np.mean([m['total_throughput'] for m in adaptive_results])
        adaptive_wait_time = np.mean([m['avg_wait_time'] for m in adaptive_results])
        adaptive_queue_length = np.mean([m['avg_queue_length'] for m in adaptive_results])
        adaptive_speed = np.mean([m['avg_speed'] for m in adaptive_results])
        
        print(f"Fixed Timing Results:")
        print(f"  Throughput: {fixed_throughput:.1f}")
        print(f"  Avg Wait Time: {fixed_wait_time:.2f}s")
        print(f"  Avg Queue Length: {fixed_queue_length:.2f}")
        print(f"  Avg Speed: {fixed_speed:.2f}m/s")
        
        print(f"\nAdaptive Timing Results:")
        print(f"  Throughput: {adaptive_throughput:.1f}")
        print(f"  Avg Wait Time: {adaptive_wait_time:.2f}s")
        print(f"  Avg Queue Length: {adaptive_queue_length:.2f}")
        print(f"  Avg Speed: {adaptive_speed:.2f}m/s")
        
        # Calculate improvements
        throughput_improvement = ((adaptive_throughput - fixed_throughput) / fixed_throughput) * 100
        wait_time_improvement = ((fixed_wait_time - adaptive_wait_time) / fixed_wait_time) * 100
        queue_improvement = ((fixed_queue_length - adaptive_queue_length) / fixed_queue_length) * 100
        speed_improvement = ((adaptive_speed - fixed_speed) / fixed_speed) * 100
        
        print(f"\nAdaptive vs Fixed Improvements:")
        print(f"  Throughput: {throughput_improvement:+.1f}%")
        print(f"  Wait Time: {wait_time_improvement:+.1f}% (reduction)")
        print(f"  Queue Length: {queue_improvement:+.1f}% (reduction)")
        print(f"  Speed: {speed_improvement:+.1f}%")
        
        # Create comparison visualization
        create_baseline_comparison_plot(fixed_results, adaptive_results)
        
    return fixed_results, adaptive_results

def create_baseline_comparison_plot(fixed_results, adaptive_results):
    """Create comparison visualization between fixed and adaptive baselines"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Fixed vs Adaptive Baseline Comparison', fontsize=16)
    
    # Extract metrics
    fixed_metrics = {
        'throughput': [m['total_throughput'] for m in fixed_results],
        'wait_time': [m['avg_wait_time'] for m in fixed_results],
        'queue_length': [m['avg_queue_length'] for m in fixed_results],
        'speed': [m['avg_speed'] for m in fixed_results]
    }
    
    adaptive_metrics = {
        'throughput': [m['total_throughput'] for m in adaptive_results],
        'wait_time': [m['avg_wait_time'] for m in adaptive_results],
        'queue_length': [m['avg_queue_length'] for m in adaptive_results],
        'speed': [m['avg_speed'] for m in adaptive_results]
    }
    
    # Box plots for comparison
    metrics_names = ['Throughput', 'Avg Wait Time (s)', 'Avg Queue Length', 'Avg Speed (m/s)']
    metric_keys = ['throughput', 'wait_time', 'queue_length', 'speed']
    
    for i, (name, key) in enumerate(zip(metrics_names, metric_keys)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        data_to_plot = [fixed_metrics[key], adaptive_metrics[key]]
        box_plot = ax.boxplot(data_to_plot, labels=['Fixed', 'Adaptive'], patch_artist=True)
        
        # Color the boxes
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        
        # Add mean values as text
        fixed_mean = np.mean(fixed_metrics[key])
        adaptive_mean = np.mean(adaptive_metrics[key])
        ax.text(1, fixed_mean, f'{fixed_mean:.1f}', ha='center', va='bottom', fontweight='bold')
        ax.text(2, adaptive_mean, f'{adaptive_mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plot saved as 'baseline_comparison.png'")

def run_single_baseline(control_type="fixed", num_runs=1):
    """Run a single baseline controller for testing"""
    controller = BaselineTrafficLight(control_type=control_type)
    results = controller.run_baseline(num_runs=num_runs, save_results=True)
    return results

# --- Main Execution ---
if __name__ == "__main__":
    print("Traffic Light Baseline Controller")
    print("=" * 50)
    
    # Menu for user selection
    print("\nSelect operation:")
    print("1. Run Fixed Timing Baseline (3 runs)")
    print("2. Run Adaptive Timing Baseline (3 runs)")
    print("3. Compare Fixed vs Adaptive (3 runs each)")
    print("4. Quick Test - Fixed (1 run)")
    print("5. Quick Test - Adaptive (1 run)")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\nRunning Fixed Timing Baseline...")
            run_single_baseline(control_type="fixed", num_runs=3)
            
        elif choice == "2":
            print("\nRunning Adaptive Timing Baseline...")
            run_single_baseline(control_type="adaptive", num_runs=3)
            
        elif choice == "3":
            print("\nRunning Comparison Analysis...")
            compare_baselines()
            
        elif choice == "4":
            print("\nRunning Quick Fixed Test...")
            run_single_baseline(control_type="fixed", num_runs=1)
            
        elif choice == "5":
            print("\nRunning Quick Adaptive Test...")
            run_single_baseline(control_type="adaptive", num_runs=1)
            
        else:
            print("Invalid choice. Running default fixed timing baseline...")
            run_single_baseline(control_type="fixed", num_runs=1)
            
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Running default fixed timing baseline...")
        run_single_baseline(control_type="fixed", num_runs=1)
    
    print("\nExecution completed.")