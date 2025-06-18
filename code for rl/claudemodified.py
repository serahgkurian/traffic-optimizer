import numpy as np
import traci
from collections import deque
import random
import tensorflow as tf
import csv
import os
from generate_routes import generate_random_routes

# Hyperparameters
CONTROLLED_LANES = ['-north_0', '-north_1', '-east_0', '-east_1', '-east_2', 
                   '-south_0', '-south_1', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
STATE_SIZE = len(CONTROLLED_LANES) + 4 + 1  # queue_lengths + phase_one_hot + phase_duration
ACTION_SIZE = 4
GAMMA = 0.95
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPISODES = 300
SUMO_BINARY = "sumo"
VEHICLES_PER_RUN = 300
MAX_STEPS = 500
TARGET_UPDATE_FREQ = 100  # Update target network every 100 steps
TRAIN_FREQ = 4  # Train every 4 steps

# --- Improved DQN Agent with Target Network ---
class ImprovedDQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.main_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.step_count = 0

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=STATE_SIZE, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(ACTION_SIZE, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA))
        return model

    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.main_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.main_model.predict(state[np.newaxis, :], verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])
        
        # Current Q values from main network
        current_q_values = self.main_model.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(BATCH_SIZE):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])
        
        # Train the main model
        self.main_model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.update_target_model()

# --- Helper Functions ---
def get_state(tls_id, controlled_lanes, last_phase, last_phase_time, sim_step):
    """Get normalized state representation"""
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes]
    current_phase = traci.trafficlight.getPhase(tls_id)
    phase_one_hot = [1 if i == current_phase else 0 for i in range(ACTION_SIZE)]

    if current_phase == last_phase:
        phase_duration = sim_step - last_phase_time
    else:
        phase_duration = 0
        last_phase_time = sim_step
        last_phase = current_phase

    # Normalize queue lengths (assuming max 20 vehicles per lane)
    normalized_queues = [min(q/20.0, 1.0) for q in queue_lengths]
    
    # Normalize phase duration (assuming max 60 seconds)
    normalized_duration = min(phase_duration/60.0, 1.0)

    state = np.array(normalized_queues + phase_one_hot + [normalized_duration], dtype=np.float32)
    return state, last_phase, last_phase_time

def compute_improved_reward(controlled_lanes, prev_total_waiting, prev_cumulative_throughput, 
                          current_phase, prev_phase, min_phase_duration=5):
    """Improved reward function focusing on traffic flow efficiency"""
    
    # Current metrics
    current_total_waiting = sum(traci.vehicle.getWaitingTime(vid) 
                               for lane in controlled_lanes 
                               for vid in traci.lane.getLastStepVehicleIDs(lane))
    
    # Get vehicles that arrived THIS step (incremental throughput)
    step_throughput = len(traci.simulation.getArrivedIDList())
    current_cumulative_throughput = prev_cumulative_throughput + step_throughput
    
    current_stopped = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
    
    # Reward components
    # 1. Throughput reward (positive reward for vehicles that completed their journey this step)
    throughput_reward = 2.0 * step_throughput
    
    # 2. Waiting time reduction (positive reward for reducing total waiting time)
    waiting_reward = 0.1 * (prev_total_waiting - current_total_waiting)
    
    # 3. Queue length penalty (negative reward for stopped vehicles)
    queue_penalty = -0.5 * current_stopped
    
    # 4. Phase change penalty (discourage frequent switching)
    phase_change_penalty = -2.0 if current_phase != prev_phase else 0.0
    
    # 5. Long waiting penalty (severe penalty for vehicles waiting too long)
    long_wait_penalty = 0
    for lane in controlled_lanes:
        for vid in traci.lane.getLastStepVehicleIDs(lane):
            wait_time = traci.vehicle.getWaitingTime(vid)
            if wait_time > 60:  # 60 seconds
                long_wait_penalty -= 5.0
            elif wait_time > 30:  # 30 seconds
                long_wait_penalty -= 1.0
    
    total_reward = (throughput_reward + waiting_reward + queue_penalty + 
                   phase_change_penalty + long_wait_penalty)
    
    return total_reward, current_total_waiting, current_cumulative_throughput, step_throughput

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
csv_path = "logs/improved_training_metrics.csv"
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "TotalReward", "AvgWaitingTime", "AvgQueueLength", 
                    "Throughput", "Epsilon"])

# --- Training Loop ---
agent = ImprovedDQNAgent()

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
    prev_total_waiting = 0
    cumulative_throughput = 0  # This will track the total vehicles that completed their journey

    state, last_phase, last_phase_time = get_state(TLS_ID, CONTROLLED_LANES, 
                                                  last_phase, last_phase_time, step)
    wait_times, queue_lengths, step_throughputs = [], [], []

    while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
        action = agent.act(state, EPSILON)
        traci.trafficlight.setPhase(TLS_ID, action)
        traci.simulationStep()
        step += 1

        next_state, last_phase, last_phase_time = get_state(TLS_ID, CONTROLLED_LANES, 
                                                           last_phase, last_phase_time, step)
        current_phase = traci.trafficlight.getPhase(TLS_ID)

        # Calculate reward using improved function
        reward, current_total_waiting, cumulative_throughput, step_throughput = compute_improved_reward(
            CONTROLLED_LANES, prev_total_waiting, cumulative_throughput, 
            current_phase, last_phase
        )

        # Collect metrics
        total_vehicles = sum(len(traci.lane.getLastStepVehicleIDs(lane)) 
                           for lane in CONTROLLED_LANES)
        avg_wait = current_total_waiting / max(total_vehicles, 1)
        queue_count = sum(traci.lane.getLastStepHaltingNumber(lane) 
                         for lane in CONTROLLED_LANES)

        total_reward += reward
        wait_times.append(avg_wait)
        queue_lengths.append(queue_count)
        step_throughputs.append(step_throughput)

        # Store experience
        done = (step >= MAX_STEPS or traci.simulation.getMinExpectedNumber() == 0)
        agent.remember(state, action, reward, next_state, done)
        
        # Train more frequently
        if step % TRAIN_FREQ == 0:
            agent.replay()

        state = next_state
        prev_total_waiting = current_total_waiting
        last_phase = current_phase

        if done:
            break

    traci.close()

    # Episode metrics
    avg_wait = np.mean(wait_times) if wait_times else 0
    avg_queue = np.mean(queue_lengths) if queue_lengths else 0
    total_step_throughput = sum(step_throughputs)  # Total vehicles that completed journey

    print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f} | "
          f"AvgWait: {avg_wait:.2f} | AvgQueue: {avg_queue:.2f} | "
          f"Throughput: {cumulative_throughput} | StepThroughput: {total_step_throughput} | Epsilon: {EPSILON:.3f}")

    # Decay epsilon
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    # Log metrics
    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode + 1, total_reward, avg_wait, avg_queue, 
                        cumulative_throughput, EPSILON])

# Save models
agent.main_model.save("logs/improved_dqn_main_model.h5")
agent.target_model.save("logs/improved_dqn_target_model.h5")
print("Models saved to logs/")