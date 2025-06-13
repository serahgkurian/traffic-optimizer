import os
import random
import numpy as np
import traci
import tensorflow as tf
from collections import deque

# ----------- Hyperparameters -----------
STATE_SIZE = 6
ACTION_SIZE = 14  # number of traffic light positions
NUM_CLASSES = 3   # G, y, r
GAMMA = 0.95
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 25

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
            tf.keras.layers.Dense(64, input_shape=(STATE_SIZE,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(ACTION_SIZE * NUM_CLASSES)  # 3 logits per signal
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA))
        return model

    def remember(self, state, action_str, reward, next_state, done):
        self.memory.append((state, action_str, reward, next_state, done))

    def act(self, state, epsilon):
        logits = self.model.predict(np.array([state]), verbose=0)[0]
        action = []
        for i in range(ACTION_SIZE):
            group = logits[i*3:(i+1)*3]
            if np.random.rand() <= epsilon:
                idx = random.choice([0, 1, 2])
            else:
                idx = np.argmax(group)
            action.append(['G', 'y', 'r'][idx])
        return ''.join(action)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action_str, reward, next_state, done in minibatch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            next_q = self.model.predict(np.array([next_state]), verbose=0)[0]

            for i in range(ACTION_SIZE):
                idx = ['G', 'y', 'r'].index(action_str[i])
                group_start = i * NUM_CLASSES
                if done:
                    target[group_start + idx] = reward
                else:
                    future_group = next_q[group_start:group_start+NUM_CLASSES]
                    target[group_start + idx] = reward + GAMMA * np.max(future_group)

            self.model.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)

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
    return -total_wait

# ----------- Training Loop -----------
agent = DQNAgent()

for episode in range(EPISODES):
    SUMO_BINARY = "sumo-gui" if episode == EPISODES else "sumo"
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    step = 0
    total_reward = 0
    done = False
    state = get_state()

    while traci.simulation.getMinExpectedNumber() > 0 and step < 1000:
        action_str = agent.act(state, EPSILON)
        traci.trafficlight.setRedYellowGreenState(TLS_ID, action_str)
        traci.simulationStep()

        next_state = get_state()
        reward = compute_reward()
        total_reward += reward
        done = traci.simulation.getMinExpectedNumber() == 0

        agent.remember(state, action_str, reward, next_state, done)
        state = next_state
        step += 1

    traci.close()
    print(f"Episode {episode+1}/{EPISODES}, Reward: {total_reward:.2f}, Steps: {step}")

    agent.replay()

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY