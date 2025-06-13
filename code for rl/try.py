import os
import random
import numpy as np
import traci
import tensorflow as tf
from collections import deque

# ----------- Hyperparameters -----------
STATE_SIZE = 6
ACTION_SIZE = 14  # number of traffic light signals
GAMMA = 0.95
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 100

# ----------- SUMO Setup -----------
SUMO_BINARY = "sumo"  # or "sumo-gui"
SUMO_CONFIG = "../trafficinter.sumocfg"
TLS_ID = "J0"
CONTROLLED_LANES = [
    '-north_0', '-north_0', '-north_1',
    '-east_0', '-east_0', '-east_1', '-east_2',
    '-south_0', '-south_0', '-south_1',
    '-west_0', '-west_0', '-west_1', '-west_2'
]

# ----------- DQN Agent -----------
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_dim=STATE_SIZE, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(ACTION_SIZE, activation='sigmoid')  # for G/y/r
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA))
        return model

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.rand(ACTION_SIZE)  # continuous random values [0, 1]
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return q_values

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for s, a, r, s_, done in minibatch:
            target = r
            if not done:
                future = self.model.predict(np.array([s_]), verbose=0)[0]
                target += GAMMA * np.max(future)
            target_vector = self.model.predict(np.array([s]), verbose=0)[0]
            # Just update the full vector toward target in the best direction
            target_vector = a  # learning to replicate the action directly
            self.model.fit(np.array([s]), np.array([target_vector]), epochs=1, verbose=0)

# ----------- Helper Functions -----------
def get_state():
    state = []
    for lane in ["-east_0", "-west_0", "-north_0", "-south_0", "-east_1", "-west_1"]:
        q = traci.lane.getLastStepHaltingNumber(lane)
        state.append(q)
    return np.array(state, dtype=np.float32)

def compute_reward():
    total_wait = 0
    for lane in CONTROLLED_LANES:
        vehs = traci.lane.getLastStepVehicleIDs(lane)
        for v in vehs:
            total_wait += traci.vehicle.getWaitingTime(v)
    return -total_wait  # lower wait = better

def action_to_tls_string(action_vector):
    state_str = ""
    for val in action_vector:
        if val > 0.66:
            state_str += 'G'
        elif val > 0.33:
            state_str += 'y'
        else:
            state_str += 'r'
    return state_str

# ----------- Training Loop -----------
agent = DQNAgent()

for episode in range(EPISODES):
    if episode % 10 == 0:
        SUMO_BINARY="sumo-gui"
    else:
        SUMO_BINARY="sumo"
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    step = 0
    total_reward = 0
    done = False
    state = get_state()

    while traci.simulation.getMinExpectedNumber() > 0 and step < 1000:
        action_vector = agent.act(state, EPSILON)
        tls_state = action_to_tls_string(action_vector)
        traci.trafficlight.setRedYellowGreenState(TLS_ID, tls_state)
        traci.simulationStep()

        next_state = get_state()
        reward = compute_reward()
        total_reward += reward
        done = traci.simulation.getMinExpectedNumber() == 0

        agent.remember(state, action_vector, reward, next_state, done)
        state = next_state
        step += 1

    traci.close()
    print(f"Episode {episode+1}/{EPISODES}, Reward: {total_reward:.2f}, Steps: {step}")

    agent.replay()

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
