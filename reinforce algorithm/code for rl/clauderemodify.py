import numpy as np
import tensorflow as tf
import traci
import os
import random
import csv
from collections import deque
from generate_routes import generate_random_routes

# Constants
PHASES = [0, 1, 2, 3]
STATE_SIZE = 14 + 1  # 14 lanes + current phase
GAMMA = 0.99
LEARNING_RATE = 0.001
EPISODES = 200
MAX_STEPS = 800
CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
                    '-east_0', '-east_0', '-east_1', '-east_2',
                    '-south_0', '-south_0', '-south_1',
                    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
VEHICLES_PER_RUN = 500
SUMO_BINARY = "sumo"
YELLOW_PHASE_OFFSET = 4
YELLOW_DURATION = 4

# --- Policy Network ---
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # outputs a value between 0 and 1

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.output_layer(x)

# --- Helper Functions ---
def get_state(current_phase):
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in CONTROLLED_LANES]
    return np.array(queue_lengths + [current_phase], dtype=np.float32)

def get_green_lanes(current_phase):
    """Get the lanes that have green light for the current phase"""
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
    return green_lane_ids

def compute_enhanced_reward(current_phase, phase_duration, vehicles_passed_this_phase):
    """
    Enhanced reward function emphasizing throughput and considering waiting times
    """
    green_lane_ids = get_green_lanes(current_phase)
    
    # Component 1: Throughput reward (primary objective)
    # Ensure non-negative throughput
    vehicles_passed = max(0, vehicles_passed_this_phase)
    throughput_reward = vehicles_passed * 12.0  # High weight for vehicles that passed
    
    # Component 2: Green lane efficiency - vehicles moving vs stopped
    green_moving_vehicles = 0
    green_stopped_vehicles = 0
    green_total_wait_time = 0
    
    for lane in green_lane_ids:
        lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
        for veh_id in lane_vehicles:
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                wait_time = traci.vehicle.getWaitingTime(veh_id)
                green_total_wait_time += wait_time
                
                if speed > 0.1:
                    green_moving_vehicles += 1
                else:
                    green_stopped_vehicles += 1
            except traci.exceptions.TraCIException:
                # Vehicle might have left during the step
                continue
    
    # Reward for vehicles moving in green lanes, penalty for stopped vehicles
    green_efficiency_reward = green_moving_vehicles * 2.0 - green_stopped_vehicles * 1.0
    
    # Component 3: Waiting time penalty (for all controlled lanes)
    total_wait_time = 0
    total_vehicles = 0
    red_lane_wait_penalty = 0
    
    for lane in CONTROLLED_LANES:
        lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
        for veh_id in lane_vehicles:
            try:
                wait_time = traci.vehicle.getWaitingTime(veh_id)
                total_wait_time += wait_time
                total_vehicles += 1
                
                # Extra penalty for vehicles waiting in red lanes
                if lane not in green_lane_ids and wait_time > 10:  # More than 10 seconds waiting
                    red_lane_wait_penalty += (wait_time - 10) * 0.1
            except traci.exceptions.TraCIException:
                # Vehicle might have left during the step
                continue
    
    # Average waiting time penalty
    avg_wait_penalty = (total_wait_time / max(total_vehicles, 1)) * 0.5
    
    # Component 4: Queue length considerations
    red_lane_queue_penalty = 0
    green_lane_queue_length = 0
    
    for lane in CONTROLLED_LANES:
        try:
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            if lane in green_lane_ids:
                green_lane_queue_length += queue_length
            else:
                # Penalty increases quadratically with queue length in red lanes
                red_lane_queue_penalty += queue_length * queue_length * 0.05
        except traci.exceptions.TraCIException:
            continue
    
    # Component 5: Phase duration efficiency
    min_efficient_duration = 8
    max_efficient_duration = 25
    
    duration_penalty = 0
    # If there are vehicles in green lanes and we're switching too early
    if green_lane_queue_length > 0 and phase_duration < min_efficient_duration:
        duration_penalty = (min_efficient_duration - phase_duration) * green_lane_queue_length * 0.5
    elif phase_duration > max_efficient_duration:
        duration_penalty = (phase_duration - max_efficient_duration) * 1.5
    
    # Component 6: Starvation prevention - reward for serving high-wait-time directions
    max_wait_in_direction = {}
    for lane in CONTROLLED_LANES:
        direction = lane.split('_')[0]  # Extract direction (e.g., '-north')
        max_wait = 0
        lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
        for veh_id in lane_vehicles:
            try:
                wait_time = traci.vehicle.getWaitingTime(veh_id)
                max_wait = max(max_wait, wait_time)
            except traci.exceptions.TraCIException:
                continue
        max_wait_in_direction[direction] = max_wait
    
    # Reward for serving the direction with highest waiting time
    current_direction_wait = 0
    for lane in green_lane_ids:
        if lane in CONTROLLED_LANES:
            direction = lane.split('_')[0]
            current_direction_wait = max(current_direction_wait, max_wait_in_direction.get(direction, 0))
    
    starvation_prevention_reward = 0
    if max_wait_in_direction:
        max_overall_wait = max(max_wait_in_direction.values())
        if max_overall_wait > 0:
            starvation_prevention_reward = (current_direction_wait / max_overall_wait) * 3.0
    
    # Final reward calculation
    total_reward = (throughput_reward +                    # Primary: vehicles passed
                   green_efficiency_reward +               # Green lane utilization
                   starvation_prevention_reward -          # Serve high-wait directions
                   avg_wait_penalty -                      # General waiting time penalty
                   red_lane_wait_penalty -                 # Extra penalty for long waits in red
                   red_lane_queue_penalty -                # Quadratic penalty for red queues
                   duration_penalty)                       # Phase timing penalty
    
    return total_reward, {
        'throughput': throughput_reward,
        'green_efficiency': green_efficiency_reward,
        'wait_penalty': -(avg_wait_penalty + red_lane_wait_penalty),
        'queue_penalty': -red_lane_queue_penalty,
        'duration_penalty': -duration_penalty,
        'starvation_prevention': starvation_prevention_reward
    }

def get_avg_wait_and_queue():
    """Get average waiting time and queue length across all controlled lanes"""
    total_wait = 0
    total_queue = 0
    total_vehicles = 0
    for lane in CONTROLLED_LANES:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)
        total_queue += sum(1 for v in veh_ids if traci.vehicle.getSpeed(v) < 0.1)
        for v in veh_ids:
            total_wait += traci.vehicle.getWaitingTime(v)
        total_vehicles += len(veh_ids)
    avg_wait = total_wait / total_vehicles if total_vehicles > 0 else 0
    return avg_wait, total_queue / len(CONTROLLED_LANES)

# --- Enhanced REINFORCE Training ---
def train(policy_net, optimizer, states, durations, rewards):
    """Enhanced training function with reward normalization"""
    # Normalize rewards to prevent exploding gradients
    rewards_array = np.array(rewards)
    if len(rewards_array) > 1:
        rewards_normalized = (rewards_array - np.mean(rewards_array)) / (np.std(rewards_array) + 1e-8)
    else:
        rewards_normalized = rewards_array
    
    # Compute discounted rewards
    discounted_rewards = []
    cumulative = 0
    for r in reversed(rewards_normalized):
        cumulative = r + GAMMA * cumulative
        discounted_rewards.insert(0, cumulative)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

    with tf.GradientTape() as tape:
        predicted_durations = tf.stack([policy_net(tf.expand_dims(s, axis=0))[0][0] for s in states])
        predicted_durations = predicted_durations * 20 + 8  # Map to range [8, 28] seconds
        
        # Use Gaussian-like log probability for continuous action
        duration_targets = tf.convert_to_tensor(durations, dtype=tf.float32)
        log_probs = -0.5 * tf.square((predicted_durations - duration_targets) / 3.0)  # Sigma = 3
        loss = -tf.reduce_sum(log_probs * discounted_rewards)

    grads = tape.gradient(loss, policy_net.trainable_variables)
    # Gradient clipping to prevent exploding gradients
    grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

# --- Main Loop ---
policy_net = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

os.makedirs("logs", exist_ok=True)
with open("logs/episode_metrics.csv", "w", newline="") as f:
    csv.writer(f).writerow(["Episode", "TotalReward", "Throughput", "AvgWaitTime", "AvgQueueLength", 
                           "ThroughputReward", "WaitPenalty", "QueuePenalty", "DurationPenalty"])

for episode in range(EPISODES):
    route_file = r"../code for rl/random_routes.rou.xml"
    generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)

    if episode == EPISODES - 1:
        SUMO_BINARY = "sumo-gui"
    traci.start([SUMO_BINARY, "-c", "../trafficinter.sumocfg"])

    current_phase = 0
    step = 0
    states, durations_taken, episode_rewards = [], [], []
    total_reward = 0
    total_throughput = 0
    wait_time_accum = []
    queue_len_accum = []
    
    # Track reward components for analysis
    reward_components = {
        'throughput': 0, 'green_efficiency': 0, 'wait_penalty': 0,
        'queue_penalty': 0, 'duration_penalty': 0, 'starvation_prevention': 0
    }

    while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
        state = get_state(current_phase)
        
        # Get duration from policy network (8-48 seconds range)
        normalized_duration = policy_net(tf.convert_to_tensor([state], dtype=tf.float32))[0].numpy()[0]
        duration = int(normalized_duration * 40 + 8)  # Scale to [8, 48] seconds
        

        # Apply green phase and track vehicles passed
        traci.trafficlight.setPhase(TLS_ID, current_phase)
        vehicles_passed_this_phase = 0
        
        for phase_step in range(duration):
            # Count vehicles that arrive (complete their journey) this simulation step
            arrived_this_step = traci.simulation.getArrivedIDList()
            vehicles_passed_this_phase += len(arrived_this_step)
            
            traci.simulationStep()
            step += 1
            if step >= MAX_STEPS: break
        
        # Apply yellow phase after green
        yellow_phase = current_phase + YELLOW_PHASE_OFFSET
        traci.trafficlight.setPhase(TLS_ID, yellow_phase)
        for _ in range(YELLOW_DURATION):
            # Count vehicles completing during yellow phase too
            arrived_this_step = traci.simulation.getArrivedIDList()
            total_throughput += len(arrived_this_step)
            
            traci.simulationStep()
            step += 1
            if step >= MAX_STEPS: break


        if step >= MAX_STEPS: break

        total_throughput += vehicles_passed_this_phase

        # Calculate enhanced reward for this phase
        phase_reward, components = compute_enhanced_reward(current_phase, duration, vehicles_passed_this_phase)
        total_reward += phase_reward
        
        # Accumulate reward components for analysis
        for key, value in components.items():
            reward_components[key] += value

        # Collect metrics
        avg_wait, avg_queue = get_avg_wait_and_queue()
        wait_time_accum.append(avg_wait)
        queue_len_accum.append(avg_queue)

        # Store for training
        states.append(tf.convert_to_tensor(state))
        durations_taken.append(duration)
        episode_rewards.append(phase_reward)

        current_phase = (current_phase + 1) % 4

    traci.close()

    # Calculate episode averages
    avg_wait = np.mean(wait_time_accum) if wait_time_accum else 0
    avg_queue = np.mean(queue_len_accum) if queue_len_accum else 0

    print(f"Episode {episode + 1}/{EPISODES} | Total Reward: {total_reward:.2f} | "
          f"Throughput: {total_throughput} | AvgWait: {avg_wait:.2f} | AvgQueue: {avg_queue:.2f}")
    print(f"  Phases completed: {len(states)}, Avg phase duration: {np.mean(durations_taken):.1f}s")
    print(f"  Reward breakdown - Throughput: {reward_components['throughput']:.1f}, "
          f"Wait Penalty: {reward_components['wait_penalty']:.1f}, "
          f"Queue Penalty: {reward_components['queue_penalty']:.1f}")
    
    # Train the policy network
    if len(states) > 0:  # Only train if we have data
        train(policy_net, optimizer, states, durations_taken, episode_rewards)

    # Log detailed metrics
    with open("logs/episode_metrics.csv", "a", newline="") as f:
        csv.writer(f).writerow([
            episode + 1, total_reward, total_throughput, avg_wait, avg_queue,
            reward_components['throughput'], reward_components['wait_penalty'],
            reward_components['queue_penalty'], reward_components['duration_penalty']
        ])

# Save the trained model
policy_net.save_weights("logs/policy_model_enhanced.weights.h5")
print("Model saved to logs/policy_model_enhanced.weights.h5")

# Print final training summary
print("\nTraining completed!")
print("Key improvements in this version:")
print("- Enhanced reward function with throughput emphasis")
print("- Waiting time penalties to discourage long waits")
print("- Starvation prevention for balanced service")
print("- Better phase duration optimization (8-28 seconds)")
print("- Reward normalization for stable training")