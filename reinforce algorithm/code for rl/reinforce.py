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
VEHICLES_PER_RUN = 300
SUMO_BINARY = "sumo-gui"
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

def compute_throughput():
    passed = len(traci.simulation.getArrivedIDList())
    return passed

def compute_reward(current_phase):
    green_lanes = traci.trafficlight.getControlledLanes(TLS_ID)

    # Get the signal state string (e.g., "rrrGGGrrr")
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    phase_state = logic.phases[current_phase].state

    # Identify lanes with green light in current phase
    green_lane_ids = []
    for i, signal in enumerate(phase_state):
        if signal == 'G' and i < len(green_lanes):
            green_lane_ids.append(green_lanes[i])

    # Count vehicles moving in green lanes
    moving_green = 0
    for lane in green_lane_ids:
        for veh_id in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getSpeed(veh_id) > 0.1:
                moving_green += 1

    # Count total stopped vehicles in all controlled lanes
    stopped_total = 0
    for lane in CONTROLLED_LANES:
        stopped_total += traci.lane.getLastStepHaltingNumber(lane)

    # Final reward calculation
    reward = 2.0 * moving_green - 1.0 * stopped_total
    return reward


def get_avg_wait_and_queue():
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

# --- REINFORCE Training ---
def train(policy_net, optimizer, states, durations, rewards):
    discounted_rewards = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + GAMMA * cumulative
        discounted_rewards.insert(0, cumulative)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

    with tf.GradientTape() as tape:
        predicted_durations = tf.stack([policy_net(tf.expand_dims(s, axis=0))[0][0] for s in states])
        predicted_durations = predicted_durations * 15 + 5  # map to range [5, 20]
        log_probs = -tf.square(predicted_durations - durations)  # pseudo log-prob
        loss = -tf.reduce_sum(log_probs * discounted_rewards)

    grads = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

# --- Main Loop ---
policy_net = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

os.makedirs("logs", exist_ok=True)
with open("logs/episode_metrics.csv", "w", newline="") as f:
    csv.writer(f).writerow(["Episode", "TotalReward", "Throughput", "AvgWaitTime", "AvgQueueLength"])

for episode in range(EPISODES):
    route_file = r"../code for rl/random_routes.rou.xml"
    generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)

    if episode == EPISODES - 1:
        SUMO_BINARY = "sumo-gui"
    traci.start([SUMO_BINARY, "-c", "../trafficinter.sumocfg"])

    current_phase = 0
    step = 0
    states, durations_taken, rewards = [], [], []
    total_reward = 0
    total_throughput = 0
    wait_time_accum = []
    queue_len_accum = []

    while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
        state = get_state(current_phase)
        normalized_duration = policy_net(tf.convert_to_tensor([state], dtype=tf.float32))[0].numpy()[0]
        duration = float(normalized_duration * 25 + 5)  # scale to [5, 30]
        duration = int(duration)

        # Apply yellow phase before green
        yellow_phase = current_phase + YELLOW_PHASE_OFFSET
        traci.trafficlight.setPhase(TLS_ID, yellow_phase)
        for _ in range(YELLOW_DURATION):
            passed = compute_throughput()
            total_throughput += passed
            traci.simulationStep()
            step += 1
            if step >= MAX_STEPS: break

        if step >= MAX_STEPS: break

        traci.trafficlight.setPhase(TLS_ID, current_phase)
        for _ in range(duration):
            reward = compute_reward(current_phase)
            passed = compute_throughput()
            total_reward += reward
            total_throughput += passed
            traci.simulationStep()
            step += 1
            if step >= MAX_STEPS: break

        reward = compute_reward(current_phase)
        passed = compute_throughput()
        total_reward += reward
        total_throughput += passed

        avg_wait, avg_queue = get_avg_wait_and_queue()
        wait_time_accum.append(avg_wait)
        queue_len_accum.append(avg_queue)

        states.append(tf.convert_to_tensor(state))
        durations_taken.append(duration)
        rewards.append(reward)

        current_phase = (current_phase + 1) % 4

    traci.close()

    avg_wait = np.mean(wait_time_accum)
    avg_queue = np.mean(queue_len_accum)

    print(f"Episode {episode + 1}/{EPISODES} | Total Reward: {total_reward:.2f} | Throughput: {total_throughput} | AvgWait: {avg_wait:.2f} | AvgQueue: {avg_queue:.2f}")
    train(policy_net, optimizer, states, durations_taken, rewards)

    with open("logs/episode_metrics.csv", "a", newline="") as f:
        csv.writer(f).writerow([episode + 1, total_reward, total_throughput, avg_wait, avg_queue])

policy_net.save_weights("logs/policy_model_continuous.weights.h5")
print("Model saved to logs/policy_model_continuous.h5")
