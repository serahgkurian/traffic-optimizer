import numpy as np
import traci
from collections import deque
import random
import tensorflow as tf

# Hyperparameters
STATE_SIZE = 14 + 4 + 1  # 14 controlled lanes, 4 one-hot phases, 1 time
ACTION_SIZE = 4         # number of phases, example 4 phases
GAMMA = 0.95
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 100

# Setup your lanes and TLS
CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
    '-east_0', '-east_0', '-east_1', '-east_2',
    '-south_0', '-south_0', '-south_1',
    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"

STATE_SIZE = len(CONTROLLED_LANES) + ACTION_SIZE + 1  # 14 + 4 + 1 = 19

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=STATE_SIZE, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(ACTION_SIZE, activation='linear')  # Q-values per phase
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

def get_state(tls_id, controlled_lanes, last_phase, last_phase_time, sim_step):
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes]
    current_phase = traci.trafficlight.getPhase(tls_id)
    num_phases = ACTION_SIZE
    phase_one_hot = [1 if i == current_phase else 0 for i in range(num_phases)]

    if current_phase == last_phase:
        phase_duration = sim_step - last_phase_time
    else:
        phase_duration = 0
        last_phase_time = sim_step
        last_phase = current_phase

    state = np.array(queue_lengths + phase_one_hot + [phase_duration], dtype=np.float32)
    return state, last_phase, last_phase_time

def compute_reward():
    # Example reward: number of vehicles passed - stopped vehicles
    stopped = 0
    for lane in CONTROLLED_LANES:
        stopped += traci.lane.getLastStepHaltingNumber(lane)
    # Number passed can be tracked from lane vehicle counts or detector data
    # For simplicity, use negative of stopped vehicles as reward here:
    return -stopped

agent = DQNAgent()

for episode in range(EPISODES):
    traci.start(["sumo", "-c", "C:\\github repos\traffic_optimizerDQN\trafficinter.sumocfg"])
    step = 0
    done = False
    total_reward = 0
    last_phase = -1
    last_phase_time = 0

    state, last_phase, last_phase_time = get_state(TLS_ID, CONTROLLED_LANES, last_phase, last_phase_time, step)

    while traci.simulation.getMinExpectedNumber() > 0 and step < 1000:
        action = agent.act(state, EPSILON)

        # Set TLS phase immediately (or add minimum duration control outside)
        traci.trafficlight.setPhase(TLS_ID, action)

        traci.simulationStep()
        step += 1

        next_state, last_phase, last_phase_time = get_state(TLS_ID, CONTROLLED_LANES, last_phase, last_phase_time, step)
        reward = compute_reward()
        total_reward += reward
        done = traci.simulation.getMinExpectedNumber() == 0

        agent.remember(state, action, reward, next_state, done)
        state = next_state

    traci.close()
    print(f"Episode {episode + 1}/{EPISODES} Reward: {total_reward:.2f} Steps: {step}")

    agent.replay()

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
