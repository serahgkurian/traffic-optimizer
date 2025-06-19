import numpy as np
import traci
from collections import deque
import random
import tensorflow as tf
import csv
import os
from generate_routes import generate_random_routes

# Hyperparameters
STATE_SIZE = 14 + 4 + 1
ACTION_SIZE = 4
GAMMA = 0.99  # was 0.95
ALPHA = 0.003  # was 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.99  # was 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 128  # was 32
MEMORY_SIZE = 20000  # was 2000
EPISODES = 300
SUMO_BINARY = "sumo"
VEHICLES_PER_RUN = 300
MAX_STEPS = 500

CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
    '-east_0', '-east_0', '-east_1', '-east_2',
    '-south_0', '-south_0', '-south_1',
    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
STATE_SIZE = len(CONTROLLED_LANES) + ACTION_SIZE + 1

# --- DQN Agent ---
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=STATE_SIZE, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(ACTION_SIZE, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA))
        return model

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for s, a, r, s_, done in minibatch:
            target = r
            if not done:
                future_q = self.model.predict(s_[np.newaxis, :], verbose=0)[0]
                target += GAMMA * np.max(future_q)
            q_values = self.model.predict(s[np.newaxis, :], verbose=0)[0]
            q_values[a] = target
            self.model.fit(s[np.newaxis, :], q_values[np.newaxis, :], epochs=1, verbose=0)

# --- Helper Functions ---
def get_state(tls_id, controlled_lanes, last_phase, last_phase_time, sim_step):
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes]
    current_phase = traci.trafficlight.getPhase(tls_id)
    phase_one_hot = [1 if i == current_phase else 0 for i in range(ACTION_SIZE)]

    if current_phase == last_phase:
        phase_duration = sim_step - last_phase_time
    else:
        phase_duration = 0
        last_phase_time = sim_step
        last_phase = current_phase

    state = np.array(queue_lengths + phase_one_hot + [phase_duration], dtype=np.float32)
    return state, last_phase, last_phase_time

def compute_reward(prev_stopped, prev_passed, prev_phase, current_phase, current_passed, max_wait_threshold=30.0):
    current_stopped = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in CONTROLLED_LANES)
    phase_changed = 1 if prev_phase != current_phase else 0

    long_wait_vehicles = 0
    for lane in CONTROLLED_LANES:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)
        for vid in veh_ids:
            if traci.vehicle.getWaitingTime(vid) > max_wait_threshold:
                long_wait_vehicles += 1

    reward = (
        + 3.0 * (current_passed - prev_passed)
        - 1.0 * (current_stopped - prev_stopped)
        - 1.0 * phase_changed
        - 2.0 * long_wait_vehicles
    )
    return reward, current_stopped

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
csv_path = "logs/training_metrics.csv"
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "TotalReward", "AvgWaitingTime", "AvgQueueLength", "Throughput"])

# --- Training Loop ---
agent = DQNAgent()

for episode in range(EPISODES):
    route_file = r"../code for rl/random_routes.rou.xml"
    generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)

    if episode == EPISODES - 1:
        SUMO_BINARY = "sumo-gui"

    traci.start([SUMO_BINARY, "-c", "../trafficinter.sumocfg"])
    step = 0
    total_reward = 0
    last_phase = -1
    last_phase_time = 0
    prev_passed = 0
    prev_stopped = 0
    cumulative_passed = 0

    state, last_phase, last_phase_time = get_state(TLS_ID, CONTROLLED_LANES, last_phase, last_phase_time, step)
    wait_times, queue_lengths = [], []

    while step < MAX_STEPS:
        action = agent.act(state, EPSILON)
        traci.trafficlight.setPhase(TLS_ID, action)
        traci.simulationStep()
        step += 1

        next_state, last_phase, last_phase_time = get_state(TLS_ID, CONTROLLED_LANES, last_phase, last_phase_time, step)
        current_phase = traci.trafficlight.getPhase(TLS_ID)

        passed_now = len(traci.simulation.getArrivedIDList())
        cumulative_passed += passed_now

        reward, current_stopped = compute_reward(
            prev_stopped, prev_passed, last_phase, current_phase, cumulative_passed
        )

        wait = sum(traci.vehicle.getWaitingTime(v) for lane in CONTROLLED_LANES for v in traci.lane.getLastStepVehicleIDs(lane))
        queue = sum(1 for lane in CONTROLLED_LANES for v in traci.lane.getLastStepVehicleIDs(lane) if traci.vehicle.getSpeed(v) < 0.1)

        total_reward += reward
        wait_times.append(wait)
        queue_lengths.append(queue)

        agent.remember(state, action, reward, next_state, False)
        state = next_state

        prev_stopped = current_stopped
        prev_passed = cumulative_passed
        last_phase = current_phase

    traci.close()

    avg_wait = np.mean(wait_times)
    avg_queue = np.mean(queue_lengths)
    total_throughput = cumulative_passed

    print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f} | AvgWait: {avg_wait:.2f} | AvgQueue: {avg_queue:.2f} | Throughput: {total_throughput}")

    agent.replay()
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode + 1, total_reward, avg_wait, avg_queue, total_throughput])

agent.model.save("logs/dqn_traffic_model.h5")
print("Model saved to logs/dqn_traffic_model.h5")
