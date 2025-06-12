import os
import random
import numpy as np
import traci
import tensorflow as tf
from collections import deque

# ----------- Hyperparameters -----------
STATE_SIZE = 6  # number of lanes or features
ACTION_SIZE = 4  # number of traffic light phases
GAMMA = 0.95
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 100

# ----------- DQN Agent -----------
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=STATE_SIZE, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(ACTION_SIZE, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA))
        return model

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for s, a, r, s_, done in minibatch:
            target = r
            if not done:
                target += GAMMA * np.amax(self.model.predict(np.array([s_]), verbose=0)[0])
            q_values = self.model.predict(np.array([s]), verbose=0)
            q_values[0][a] = target
            self.model.fit(np.array([s]), q_values, epochs=1, verbose=0)

# ----------- SUMO Setup -----------
SUMO_BINARY = "sumo-gui"  # or "sumo-gui"
SUMO_CONFIG = "../trafficinter.sumocfg"
TLS_ID = "J0"
PHASES = [0, 1, 2, 3]  # assume 4 phases for example
LANES = ["east_0", "west_0", "north_0", "south_0", "east_1", "west_1"]

def get_state():
    state = []
    for lane in LANES:
        queue_len = traci.lane.getLastStepHaltingNumber(lane)
        state.append(queue_len)
    return np.array(state, dtype=np.float32)

def compute_reward():
    total_wait = 0
    for lane in LANES:
        vehs = traci.lane.getLastStepVehicleIDs(lane)
        for v in vehs:
            total_wait += traci.vehicle.getWaitingTime(v)
    return -total_wait  # lower wait = better

# ----------- Main Loop -----------
agent = DQNAgent()

for episode in range(EPISODES):
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    step = 0
    total_reward = 0
    done = False
    state = get_state()
    
    while step < 500:
        action = agent.act(state, EPSILON)
        traci.trafficlight.setPhase(TLS_ID, PHASES[action])
        traci.simulationStep()
        
        next_state = get_state()
        reward = compute_reward()
        total_reward += reward
        done = step >= 499
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        step += 1
    
    traci.close()
    print(f"Episode {episode+1}/{EPISODES}, Reward: {total_reward}")
    
    agent.replay()
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
