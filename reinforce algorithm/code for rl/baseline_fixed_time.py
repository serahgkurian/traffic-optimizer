import traci
import os
import sys
import numpy as np
from generate_routes import generate_random_routes

# ------------- CONFIG ----------------
SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = r"C:\\github repos\\traffic_optimizerDQN\\reinforce algorithm\\trafficinter.sumocfg"
TLS_ID = "J0"  # Update this to match your actual traffic light ID
SIMULATION_STEPS = 1000
INCOMING_LANES = [
    '-north_0', '-north_1',
    '-east_0', '-east_1', '-east_2',
    '-south_0', '-south_1',
    '-west_0', '-west_1', '-west_2'
]
NUM_RUNS = 4
VEHICLES_PER_RUN = 100

# Traffic Light Phase Configuration
# Based on your tls.add.xml file
PHASE_CONFIG = [
    # Green phases
    {"duration": 30, "state": "rrrGGGrrrrGGGr", "name": "East-West Straight+Right"},
    {"duration": 30, "state": "rrrrrrGrrrrrrG", "name": "East-West Left"},
    {"duration": 30, "state": "GGrrrrrGGrrrrr", "name": "North-South Straight+Right"},
    {"duration": 30, "state": "rrGrrrrrrGrrrr", "name": "North-South Left"},
    # Yellow phases
    {"duration": 5, "state": "rrryyyrrrryyyr", "name": "East-West Yellow"},
    {"duration": 5, "state": "rrrrrryrrrrrry", "name": "East-West Left Yellow"},
    {"duration": 5, "state": "yyrrrrryyrrrrr", "name": "North-South Yellow"},
    {"duration": 5, "state": "rryrrrrrryrrrr", "name": "North-South Left Yellow"}
]

# Phase sequence (indices into PHASE_CONFIG)
PHASE_SEQUENCE = [0, 4, 1, 5, 2, 6, 3, 7]  # Green -> Yellow -> Green -> Yellow pattern

os.makedirs("logs", exist_ok=True)

# ----------- METRIC FUNCTIONS --------
def get_waiting_vehicles(lanes):
    """Count total halting vehicles across all lanes"""
    total_waiting = 0
    for lane in lanes:
        try:
            total_waiting += traci.lane.getLastStepHaltingNumber(lane)
        except traci.exceptions.TraCIException:
            # Lane might not exist or be accessible
            continue
    return total_waiting

def get_average_speed(lanes):
    """Calculate average speed across all lanes"""
    speeds = []
    for lane in lanes:
        try:
            speed = traci.lane.getLastStepMeanSpeed(lane)
            if speed >= 0:  # Valid speed
                speeds.append(speed)
        except traci.exceptions.TraCIException:
            continue
    return np.mean(speeds) if speeds else 0.0

def get_queue_length(lanes):
    """Count vehicles with speed < 0.1 m/s (effectively stopped)"""
    queue_count = 0
    for lane in lanes:
        try:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle_id in vehicle_ids:
                try:
                    if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                        queue_count += 1
                except traci.exceptions.TraCIException:
                    continue
        except traci.exceptions.TraCIException:
            continue
    return queue_count

def get_throughput():
    """Get number of vehicles that completed their trip this step"""
    try:
        return len(traci.simulation.getArrivedIDList())
    except traci.exceptions.TraCIException:
        return 0

# ----------- TRAFFIC LIGHT CONTROL --------
def initialize_traffic_light():
    """Initialize traffic light with our custom phases"""
    try:
        # Check if traffic light exists
        tls_list = traci.trafficlight.getIDList()
        print(f"Available traffic lights: {tls_list}")
        
        if TLS_ID not in tls_list:
            print(f"Warning: Traffic light {TLS_ID} not found. Using first available.")
            if tls_list:
                return tls_list[0]
            else:
                print("No traffic lights found in simulation!")
                return None
        
        # Set initial phase
        traci.trafficlight.setPhase(TLS_ID, 0)
        return TLS_ID
        
    except traci.exceptions.TraCIException as e:
        print(f"Error initializing traffic light: {e}")
        return None

def update_traffic_light(tls_id, step, phase_timer, current_phase_idx):
    """Update traffic light phase based on timing"""
    if tls_id is None:
        return phase_timer, current_phase_idx
    
    try:
        current_phase = PHASE_CONFIG[PHASE_SEQUENCE[current_phase_idx]]
        
        if phase_timer >= current_phase["duration"]:
            # Move to next phase
            current_phase_idx = (current_phase_idx + 1) % len(PHASE_SEQUENCE)
            next_phase = PHASE_CONFIG[PHASE_SEQUENCE[current_phase_idx]]
            
            # Update traffic light
            traci.trafficlight.setRedYellowGreenState(tls_id, next_phase["state"])
            
            print(f"Step {step}: Switching to phase {current_phase_idx} - {next_phase['name']}")
            phase_timer = 0
        
        return phase_timer + 1, current_phase_idx
        
    except traci.exceptions.TraCIException as e:
        print(f"Error updating traffic light: {e}")
        return phase_timer + 1, current_phase_idx

# ----------- MAIN SIMULATION LOOP --------
def run_single_simulation(run_id):
    """Run a single simulation with traffic light control"""
    SUMO_BINARY_RUN = "sumo-gui" if run_id == 0 else "sumo"
    
    # Generate random routes
    route_file = r"C:\\github repos\\traffic_optimizerDQN\\reinforce algorithm\\code for rl\\random_routes.rou.xml"
    generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)

    try:
        traci.start([SUMO_BINARY_RUN, "-c", SUMO_CONFIG])
        print(f"Run {run_id + 1}: Simulation started...")

        # Initialize traffic light
        active_tls_id = initialize_traffic_light()
        
        # Metrics storage
        metrics = {
            'waiting_vehicles': [],
            'avg_speeds': [],
            'queue_lengths': [],
            'throughputs': [],
            'phases': []
        }
        
        step = 0
        phase_timer = 0
        current_phase_idx = 0

        while traci.simulation.getMinExpectedNumber() > 0 and step < SIMULATION_STEPS:
            # Update traffic light
            phase_timer, current_phase_idx = update_traffic_light(
                active_tls_id, step, phase_timer, current_phase_idx
            )
            
            # Advance simulation
            traci.simulationStep()

            # Collect metrics
            wait = get_waiting_vehicles(INCOMING_LANES)
            speed = get_average_speed(INCOMING_LANES)
            queue = get_queue_length(INCOMING_LANES)
            throughput = get_throughput()

            metrics['waiting_vehicles'].append(wait)
            metrics['avg_speeds'].append(speed)
            metrics['queue_lengths'].append(queue)
            metrics['throughputs'].append(throughput)
            metrics['phases'].append(current_phase_idx)

            step += 1

            # Progress reporting
            if step % 100 == 0:
                print(f"  Step {step}: Wait={wait}, Speed={speed:.2f}, Queue={queue}, Throughput={throughput}")

    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        traci.close()
    
    print(f"Run {run_id + 1}: Finished at step {step}.\n")

    # Save detailed run log
    log_path = f"logs/run_{run_id+1}_detailed_metrics.csv"
    with open(log_path, "w") as f:
        f.write("step,waiting_vehicles,avg_speed,queue_length,throughput,phase_index,phase_name\n")
        for i in range(len(metrics['waiting_vehicles'])):
            phase_idx = metrics['phases'][i]
            phase_name = PHASE_CONFIG[PHASE_SEQUENCE[phase_idx]]['name']
            f.write(f"{i},{metrics['waiting_vehicles'][i]},{metrics['avg_speeds'][i]:.2f},"
                   f"{metrics['queue_lengths'][i]},{metrics['throughputs'][i]},{phase_idx},{phase_name}\n")

    # Calculate summary statistics
    return {
        "wait_sum": np.sum(metrics['waiting_vehicles']),
        "wait_avg": np.mean(metrics['waiting_vehicles']),
        "wait_max": np.max(metrics['waiting_vehicles']),
        "speed_avg": np.mean(metrics['avg_speeds']),
        "speed_min": np.min(metrics['avg_speeds']),
        "queue_avg": np.mean(metrics['queue_lengths']),
        "queue_max": np.max(metrics['queue_lengths']),
        "throughput_total": np.sum(metrics['throughputs']),
        "steps_completed": len(metrics['waiting_vehicles'])
    }

# ----------- MULTI-RUN WRAPPER --------
def run_multiple_simulations():
    """Run multiple simulations and aggregate results"""
    all_metrics = []
    
    print("=== Starting Baseline Traffic Light Simulation ===")
    print(f"Configuration:")
    print(f"  - Number of runs: {NUM_RUNS}")
    print(f"  - Vehicles per run: {VEHICLES_PER_RUN}")
    print(f"  - Max simulation steps: {SIMULATION_STEPS}")
    print(f"  - Traffic light phases: {len(PHASE_CONFIG)}")
    print()

    for run_id in range(NUM_RUNS):
        print(f"Starting run {run_id + 1}/{NUM_RUNS}...")
        summary = run_single_simulation(run_id)
        all_metrics.append(summary)

    # Calculate overall statistics
    print("\n=== Overall Summary Across All Runs ===")
    
    metrics_summary = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        metrics_summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    # Print summary
    print(f"Average Waiting Vehicles/Step: {metrics_summary['wait_avg']['mean']:.2f} ± {metrics_summary['wait_avg']['std']:.2f}")
    print(f"Maximum Waiting Vehicles: {metrics_summary['wait_max']['mean']:.2f} ± {metrics_summary['wait_max']['std']:.2f}")
    print(f"Average Speed (m/s): {metrics_summary['speed_avg']['mean']:.2f} ± {metrics_summary['speed_avg']['std']:.2f}")
    print(f"Minimum Speed (m/s): {metrics_summary['speed_min']['mean']:.2f} ± {metrics_summary['speed_min']['std']:.2f}")
    print(f"Average Queue Length: {metrics_summary['queue_avg']['mean']:.2f} ± {metrics_summary['queue_avg']['std']:.2f}")
    print(f"Maximum Queue Length: {metrics_summary['queue_max']['mean']:.2f} ± {metrics_summary['queue_max']['std']:.2f}")
    print(f"Total Throughput: {metrics_summary['throughput_total']['mean']:.0f} ± {metrics_summary['throughput_total']['std']:.0f}")
    print(f"Average Steps Completed: {metrics_summary['steps_completed']['mean']:.0f}")

    # Save comprehensive summary
    with open("logs/comprehensive_summary.csv", "w") as f:
        f.write("run,wait_sum,wait_avg,wait_max,speed_avg,speed_min,queue_avg,queue_max,throughput_total,steps_completed\n")
        for i, m in enumerate(all_metrics):
            f.write(f"{i+1},{m['wait_sum']:.2f},{m['wait_avg']:.2f},{m['wait_max']:.2f},"
                   f"{m['speed_avg']:.2f},{m['speed_min']:.2f},{m['queue_avg']:.2f},"
                   f"{m['queue_max']:.2f},{m['throughput_total']:.0f},{m['steps_completed']}\n")

    # Save statistical summary
    with open("logs/statistical_summary.csv", "w") as f:
        f.write("metric,mean,std,min,max\n")
        for key, stats in metrics_summary.items():
            f.write(f"{key},{stats['mean']:.4f},{stats['std']:.4f},{stats['min']:.4f},{stats['max']:.4f}\n")

    print(f"\nResults saved to logs/ directory")
    print("Files created:")
    print("  - run_X_detailed_metrics.csv (detailed per-step data)")
    print("  - comprehensive_summary.csv (summary per run)")
    print("  - statistical_summary.csv (statistical analysis)")

# ----------- ENTRY POINT --------
if __name__ == "__main__":
    run_multiple_simulations()