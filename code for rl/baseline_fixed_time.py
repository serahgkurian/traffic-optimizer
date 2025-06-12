import traci
import os
import sys
import numpy as np
from generate_routes import generate_random_routes

# ------------- CONFIG ----------------
SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = r"../trafficinter.sumocfg"
TLS_ID = "J3"
SIMULATION_STEPS = 1000
# GREEN_DURATION = 30
INCOMING_LANES = [
    '-north_0', '-north_1',
    '-east_0', '-east_1', '-east_2',
    '-south_0', '-south_1',
    '-west_0', '-west_1', '-west_2'
]
NUM_RUNS = 5
VEHICLES_PER_RUN = 100

os.makedirs("logs", exist_ok=True)

# ----------- METRIC FUNCTIONS --------
def get_waiting_vehicles(lanes):
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)

def get_average_speed(lanes):
    speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in lanes]
    return np.mean(speeds)

def get_queue_length(lanes):
    q = 0
    for lane in lanes:
        for v in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getSpeed(v) < 0.1:
                q += 1
    return q

# ----------- MAIN BASELINE LOOP --------
def run_single_simulation(run_id):
    SUMO_BINARY = "sumo-gui" if run_id == 0 else "sumo"
    # Generate random routes
    route_file = r"../code for rl/random_routes.rou.xml"
    generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)

    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    print(f"Run {run_id + 1}: Simulation started...")

    total_wait_times, avg_speeds, queue_lengths, throughputs = [], [], [], []
    step, phase_index = 0, 0

    while traci.simulation.getMinExpectedNumber() > 0:
        # if step % GREEN_DURATION == 0:
        #     traci.trafficlight.setPhase(TLS_ID, phase_index)
        #     phase_index = (phase_index + 1) % 5

        traci.simulationStep()

        wait = get_waiting_vehicles(INCOMING_LANES)
        speed = get_average_speed(INCOMING_LANES)
        queue = get_queue_length(INCOMING_LANES)
        throughput = len(traci.simulation.getArrivedIDList())

        total_wait_times.append(wait)
        avg_speeds.append(speed)
        queue_lengths.append(queue)
        throughputs.append(throughput)

        step += 1

    traci.close()
    print(f"Run {run_id + 1}: Finished at step {step}.\n")

    # Save run log
    log_path = f"logs/run_{run_id+1}_metrics.csv"
    with open(log_path, "w") as f:
        f.write("step,waiting_vehicles,avg_speed,queue_length,throughput\n")
        for i in range(step):
            f.write(f"{i},{total_wait_times[i]},{avg_speeds[i]:.2f},{queue_lengths[i]},{throughputs[i]}\n")

    return {
        "wait_sum": np.sum(total_wait_times),
        "wait_avg": np.mean(total_wait_times),
        "speed_avg": np.mean(avg_speeds),
        "queue_avg": np.mean(queue_lengths),
        "throughput_total": np.sum(throughputs)
    }

# ----------- MULTI-RUN WRAPPER --------
def run_multiple_simulations():
    all_metrics = []

    for run_id in range(NUM_RUNS):
        summary = run_single_simulation(run_id)
        all_metrics.append(summary)

    # Summary stats
    print("\n=== Overall Summary Across Runs ===")
    avg_wait = np.mean([m["wait_avg"] for m in all_metrics])
    avg_speed = np.mean([m["speed_avg"] for m in all_metrics])
    avg_queue = np.mean([m["queue_avg"] for m in all_metrics])
    avg_throughput = np.mean([m["throughput_total"] for m in all_metrics])

    print(f"Average Waiting Vehicles/Step: {avg_wait:.2f}")
    print(f"Average Speed (m/s): {avg_speed:.2f}")
    print(f"Average Queue Length: {avg_queue:.2f}")
    print(f"Average Throughput (Vehicles Exited): {avg_throughput:.0f}")

    # Save summary CSV
    with open("logs/summary_across_runs.csv", "w") as f:
        f.write("run,wait_sum,wait_avg,speed_avg,queue_avg,throughput_total\n")
        for i, m in enumerate(all_metrics):
            f.write(f"{i+1},{m['wait_sum']:.2f},{m['wait_avg']:.2f},{m['speed_avg']:.2f},{m['queue_avg']:.2f},{m['throughput_total']:.0f}\n")

# ----------- ENTRY POINT --------
if __name__ == "__main__":
    run_multiple_simulations()
