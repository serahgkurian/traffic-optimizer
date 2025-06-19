import numpy as np
import tensorflow as tf
import traci
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from generate_routes import generate_random_routes

# Constants (same as training)
PHASES = [0, 1, 2, 3]
STATE_SIZE = 14 + 1
CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
                    '-east_0', '-east_0', '-east_1', '-east_2',
                    '-south_0', '-south_0', '-south_1',
                    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
YELLOW_PHASE_OFFSET = 4
YELLOW_DURATION = 4
MAX_STEPS = 800
VEHICLES_PER_RUN = 500

# Policy Network class (same as training)
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.output_layer(x)

def get_state(current_phase):
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in CONTROLLED_LANES]
    return np.array(queue_lengths + [current_phase], dtype=np.float32)

def compute_reward(current_phase):
    green_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    phase_state = logic.phases[current_phase].state
    links = traci.trafficlight.getControlledLinks(TLS_ID)

    green_lane_ids = []
    for i, signal in enumerate(phase_state):
        if signal == 'G':
            try:
                incoming_lane = links[i][0][0]
                green_lane_ids.append(incoming_lane)
            except:
                pass

    moving_green = 0
    for lane in green_lane_ids:
        for veh_id in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getSpeed(veh_id) > 0.1:
                moving_green += 1

    stopped_total = 0
    for lane in CONTROLLED_LANES:
        stopped_total += traci.lane.getLastStepHaltingNumber(lane)

    reward = 2.0 * moving_green - 1.0 * stopped_total
    return reward

def get_comprehensive_metrics():
    """Get detailed performance metrics"""
    metrics = {}
    
    # Throughput metrics
    metrics['vehicles_arrived'] = len(traci.simulation.getArrivedIDList())
    metrics['vehicles_departed'] = len(traci.simulation.getDepartedIDList())
    metrics['vehicles_running'] = traci.simulation.getMinExpectedNumber()
    
    # Waiting time and queue metrics
    total_wait = 0
    total_queue = 0
    total_vehicles = 0
    lane_metrics = {}
    
    for lane in CONTROLLED_LANES:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)
        lane_queue = sum(1 for v in veh_ids if traci.vehicle.getSpeed(v) < 0.1)
        lane_wait = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
        
        lane_metrics[lane] = {
            'vehicles': len(veh_ids),
            'queue': lane_queue,
            'total_wait': lane_wait,
            'avg_wait': lane_wait / len(veh_ids) if len(veh_ids) > 0 else 0
        }
        
        total_queue += lane_queue
        total_wait += lane_wait
        total_vehicles += len(veh_ids)
    
    metrics['avg_waiting_time'] = total_wait / total_vehicles if total_vehicles > 0 else 0
    metrics['avg_queue_length'] = total_queue / len(CONTROLLED_LANES)
    metrics['total_queue'] = total_queue
    metrics['lane_metrics'] = lane_metrics
    
    # Speed metrics
    all_speeds = []
    for lane in CONTROLLED_LANES:
        for veh_id in traci.lane.getLastStepVehicleIDs(lane):
            all_speeds.append(traci.vehicle.getSpeed(veh_id))
    
    metrics['avg_speed'] = np.mean(all_speeds) if all_speeds else 0
    metrics['speed_std'] = np.std(all_speeds) if all_speeds else 0
    
    return metrics

def run_single_inference(policy_net, run_id, show_gui=False):
    """Run a single inference session and collect detailed metrics"""
    
    # Generate routes
    route_file = r"../code for rl/random_routes.rou.xml"
    generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)
    
    # Start SUMO
    sumo_binary = "sumo-gui" if show_gui else "sumo"
    traci.start([sumo_binary, "-c", "../trafficinter.sumocfg"])
    
    current_phase = 0
    step = 0
    
    # Data collection
    phase_data = []
    step_metrics = []
    total_reward = 0
    phase_count = 0
    
    print(f"Run {run_id}: Starting inference...")
    
    while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
        # Get current state
        state = get_state(current_phase)
        
        # Get action from trained policy
        normalized_duration = policy_net(tf.convert_to_tensor([state], dtype=tf.float32))[0].numpy()[0]
        duration = float(normalized_duration * 40 + 8)  # scale to [8, 48]
        duration = int(max(8, min(48, duration)))
        
        phase_start_step = step
        phase_start_metrics = get_comprehensive_metrics()
        
        # Apply yellow phase
        yellow_phase = current_phase + YELLOW_PHASE_OFFSET
        traci.trafficlight.setPhase(TLS_ID, yellow_phase)
        for _ in range(YELLOW_DURATION):
            traci.simulationStep()
            step += 1
            if step >= MAX_STEPS: break
        
        if step >= MAX_STEPS: break
        
        # Apply green phase and collect metrics during the phase
        traci.trafficlight.setPhase(TLS_ID, current_phase)
        phase_rewards = []
        
        for phase_step in range(duration):
            reward = compute_reward(current_phase)
            phase_rewards.append(reward)
            total_reward += reward
            
            # Collect step-by-step metrics every 10 steps
            if step % 10 == 0:
                metrics = get_comprehensive_metrics()
                metrics['step'] = step
                metrics['phase'] = current_phase
                metrics['run_id'] = run_id
                step_metrics.append(metrics)
            
            traci.simulationStep()
            step += 1
            if step >= MAX_STEPS: break
        
        # Phase summary
        phase_end_metrics = get_comprehensive_metrics()
        phase_info = {
            'run_id': run_id,
            'phase_number': phase_count,
            'phase_id': current_phase,
            'duration': duration,
            'start_step': phase_start_step,
            'end_step': step,
            'avg_reward': np.mean(phase_rewards),
            'total_reward': sum(phase_rewards),
            'queue_start': phase_start_metrics['total_queue'],
            'queue_end': phase_end_metrics['total_queue'],
            'queue_reduction': phase_start_metrics['total_queue'] - phase_end_metrics['total_queue'],
            'throughput_during_phase': phase_end_metrics['vehicles_arrived'] - phase_start_metrics['vehicles_arrived'],
            'avg_wait_start': phase_start_metrics['avg_waiting_time'],
            'avg_wait_end': phase_end_metrics['avg_waiting_time'],
            'avg_speed': phase_end_metrics['avg_speed']
        }
        phase_data.append(phase_info)
        
        print(f"  Phase {current_phase} (#{phase_count}): {duration}s, Reward: {np.mean(phase_rewards):.2f}, "
              f"Queue: {phase_start_metrics['total_queue']}→{phase_end_metrics['total_queue']}, "
              f"Throughput: {phase_info['throughput_during_phase']}")
        
        current_phase = (current_phase + 1) % 4
        phase_count += 1
    
    # Final metrics
    final_metrics = get_comprehensive_metrics()
    final_metrics['run_id'] = run_id
    final_metrics['total_reward'] = total_reward
    final_metrics['total_phases'] = phase_count
    final_metrics['total_steps'] = step
    
    print(f"Run {run_id} completed: Total Reward: {total_reward:.2f}, "
          f"Throughput: {final_metrics['vehicles_arrived']}, "
          f"Avg Wait: {final_metrics['avg_waiting_time']:.2f}s")
    
    traci.close()
    
    return final_metrics, phase_data, step_metrics

def run_multiple_inference_sessions(num_runs=5, show_gui_last=True):
    """Run multiple inference sessions and collect comprehensive data"""
    
    # Load model
    try:
        policy_net = tf.keras.models.load_model("logs/policy_model_continuous.keras")
        print("Loaded full model successfully")
    except:
        policy_net = PolicyNetwork()
        dummy_state = tf.zeros((1, STATE_SIZE))
        _ = policy_net(dummy_state)
        policy_net.load_weights("logs/policy_model_enhanced.weights.h5")
        print("Loaded model weights successfully")
    
    # Create results directory
    os.makedirs("inference_results", exist_ok=True)
    
    all_run_data = []
    all_phase_data = []
    all_step_data = []
    
    for run_id in range(1, num_runs + 1):
        show_gui = show_gui_last and (run_id == num_runs)
        run_metrics, phase_data, step_data = run_single_inference(policy_net, run_id, show_gui)
        
        all_run_data.append(run_metrics)
        all_phase_data.extend(phase_data)
        all_step_data.extend(step_data)
    
    # Save raw data
    save_results_to_csv(all_run_data, all_phase_data, all_step_data)
    
    # Generate visualizations
    create_performance_visualizations(all_run_data, all_phase_data, all_step_data)
    
    # Print summary statistics
    print_summary_statistics(all_run_data, all_phase_data)
    
    return all_run_data, all_phase_data, all_step_data

def save_results_to_csv(run_data, phase_data, step_data):
    """Save all collected data to CSV files"""
    
    # Run summary data
    df_runs = pd.DataFrame(run_data)
    df_runs.to_csv("inference_results/run_summary.csv", index=False)
    
    # Phase-by-phase data
    df_phases = pd.DataFrame(phase_data)
    df_phases.to_csv("inference_results/phase_details.csv", index=False)
    
    # Step-by-step data (if not too large)
    if step_data:
        df_steps = pd.DataFrame(step_data)
        df_steps.to_csv("inference_results/step_metrics.csv", index=False)
    
    print("Data saved to CSV files in inference_results/")

def create_performance_visualizations(run_data, phase_data, step_data):
    """Create comprehensive performance visualizations"""
    
    df_runs = pd.DataFrame(run_data)
    df_phases = pd.DataFrame(phase_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Traffic Light RL Performance Analysis', fontsize=16)
    
    # 1. Throughput across runs
    axes[0,0].bar(df_runs['run_id'], df_runs['vehicles_arrived'])
    axes[0,0].set_title('Vehicle Throughput by Run')
    axes[0,0].set_xlabel('Run ID')
    axes[0,0].set_ylabel('Vehicles Arrived')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Average waiting time across runs
    axes[0,1].bar(df_runs['run_id'], df_runs['avg_waiting_time'])
    axes[0,1].set_title('Average Waiting Time by Run')
    axes[0,1].set_xlabel('Run ID')
    axes[0,1].set_ylabel('Avg Wait Time (s)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Total reward across runs
    axes[0,2].bar(df_runs['run_id'], df_runs['total_reward'])
    axes[0,2].set_title('Total Reward by Run')
    axes[0,2].set_xlabel('Run ID')
    axes[0,2].set_ylabel('Total Reward')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Phase duration distribution
    axes[1,0].hist(df_phases['duration'], bins=15, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Distribution of Phase Durations')
    axes[1,0].set_xlabel('Duration (s)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Queue reduction by phase
    axes[1,1].scatter(df_phases['duration'], df_phases['queue_reduction'], alpha=0.6)
    axes[1,1].set_title('Queue Reduction vs Phase Duration')
    axes[1,1].set_xlabel('Phase Duration (s)')
    axes[1,1].set_ylabel('Queue Reduction')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Performance by phase type
    phase_performance = df_phases.groupby('phase_id').agg({
        'avg_reward': 'mean',
        'queue_reduction': 'mean',
        'throughput_during_phase': 'mean'
    })
    
    x = range(len(phase_performance))
    width = 0.25
    axes[1,2].bar([i - width for i in x], phase_performance['avg_reward'], width, label='Avg Reward', alpha=0.8)
    axes[1,2].bar(x, phase_performance['queue_reduction'], width, label='Queue Reduction', alpha=0.8)
    axes[1,2].bar([i + width for i in x], phase_performance['throughput_during_phase'], width, label='Throughput', alpha=0.8)
    axes[1,2].set_title('Performance by Phase Type')
    axes[1,2].set_xlabel('Phase ID')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels([f'Phase {i}' for i in phase_performance.index])
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inference_results/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional time-series plot if step data available
    if step_data:
        create_time_series_plots(step_data)

def create_time_series_plots(step_data):
    """Create time-series visualizations"""
    df_steps = pd.DataFrame(step_data)
    
    if len(df_steps) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Time Series Performance Metrics', fontsize=16)
    
    # Plot for each run
    for run_id in df_steps['run_id'].unique():
        run_data = df_steps[df_steps['run_id'] == run_id]
        
        axes[0,0].plot(run_data['step'], run_data['vehicles_arrived'], label=f'Run {run_id}', alpha=0.7)
        axes[0,1].plot(run_data['step'], run_data['avg_waiting_time'], label=f'Run {run_id}', alpha=0.7)
        axes[1,0].plot(run_data['step'], run_data['total_queue'], label=f'Run {run_id}', alpha=0.7)
        axes[1,1].plot(run_data['step'], run_data['avg_speed'], label=f'Run {run_id}', alpha=0.7)
    
    axes[0,0].set_title('Cumulative Throughput vs Time')
    axes[0,0].set_xlabel('Simulation Step')
    axes[0,0].set_ylabel('Vehicles Arrived')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_title('Average Waiting Time vs Time')
    axes[0,1].set_xlabel('Simulation Step')
    axes[0,1].set_ylabel('Avg Wait Time (s)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].set_title('Total Queue Length vs Time')
    axes[1,0].set_xlabel('Simulation Step')
    axes[1,0].set_ylabel('Total Queue Length')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].set_title('Average Speed vs Time')
    axes[1,1].set_xlabel('Simulation Step')
    axes[1,1].set_ylabel('Avg Speed (m/s)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inference_results/time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(run_data, phase_data):
    """Print comprehensive summary statistics"""
    df_runs = pd.DataFrame(run_data)
    df_phases = pd.DataFrame(phase_data)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"\nRUN STATISTICS (across {len(df_runs)} runs):")
    print(f"Average Throughput: {df_runs['vehicles_arrived'].mean():.2f} ± {df_runs['vehicles_arrived'].std():.2f}")
    print(f"Average Wait Time: {df_runs['avg_waiting_time'].mean():.2f} ± {df_runs['avg_waiting_time'].std():.2f} seconds")
    print(f"Average Queue Length: {df_runs['avg_queue_length'].mean():.2f} ± {df_runs['avg_queue_length'].std():.2f}")
    print(f"Average Speed: {df_runs['avg_speed'].mean():.2f} ± {df_runs['speed_std'].mean():.2f} m/s")
    print(f"Total Reward: {df_runs['total_reward'].mean():.2f} ± {df_runs['total_reward'].std():.2f}")
    
    print(f"\nPHASE STATISTICS (across {len(df_phases)} phase transitions):")
    print(f"Average Phase Duration: {df_phases['duration'].mean():.2f} ± {df_phases['duration'].std():.2f} seconds")
    print(f"Average Queue Reduction per Phase: {df_phases['queue_reduction'].mean():.2f}")
    print(f"Average Throughput per Phase: {df_phases['throughput_during_phase'].mean():.2f}")
    
    print(f"\nPHASE TYPE BREAKDOWN:")
    phase_stats = df_phases.groupby('phase_id').agg({
        'duration': ['mean', 'std'],
        'avg_reward': ['mean', 'std'],
        'queue_reduction': ['mean', 'std'],
        'throughput_during_phase': ['mean', 'std']
    }).round(2)
    
    for phase_id in sorted(df_phases['phase_id'].unique()):
        print(f"  Phase {phase_id}:")
        print(f"    Duration: {phase_stats.loc[phase_id, ('duration', 'mean')]} ± {phase_stats.loc[phase_id, ('duration', 'std')]}")
        print(f"    Avg Reward: {phase_stats.loc[phase_id, ('avg_reward', 'mean')]} ± {phase_stats.loc[phase_id, ('avg_reward', 'std')]}")
        print(f"    Queue Reduction: {phase_stats.loc[phase_id, ('queue_reduction', 'mean')]} ± {phase_stats.loc[phase_id, ('queue_reduction', 'std')]}")

# Main execution
if __name__ == "__main__":
    print("Starting comprehensive inference analysis...")
    
    # Run multiple inference sessions
    num_runs = 5  # Adjust as needed
    run_data, phase_data, step_data = run_multiple_inference_sessions(
        num_runs=num_runs, 
        show_gui_last=True  # Show GUI only for the last run
    )
    
    print(f"\nAnalysis complete! Check the 'inference_results/' folder for:")
    print("- CSV files with detailed data")
    print("- Performance visualization plots")
    print("- Summary statistics printed above")