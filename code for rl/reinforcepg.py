import numpy as np
import tensorflow as tf
import traci
import os
import random
import csv
from collections import deque

# Constants
PHASES = [0, 1, 2, 3]
DURATIONS = [5, 10, 15, 20]  # seconds
ACTION_SIZE = len(DURATIONS)
STATE_SIZE = 14 + 1  # 14 lanes + current phase
GAMMA = 0.99
LEARNING_RATE = 0.001
EPISODES = 100
MAX_PHASES_PER_EPISODE = 30
CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
                    '-east_0', '-east_0', '-east_1', '-east_2',
                    '-south_0', '-south_0', '-south_1',
                    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
SUMO_BINARY = "sumo-gui"

# --- Policy Network ---
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(ACTION_SIZE, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.output_layer(x)

# --- Helper Functions ---
def get_state(current_phase):
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in CONTROLLED_LANES]
    state = np.array(queue_lengths + [current_phase], dtype=np.float32)
    return state

def compute_reward():
    passed = len(traci.simulation.getArrivedIDList())
    wait_penalty = sum(
        1 for lane in CONTROLLED_LANES for v in traci.lane.getLastStepVehicleIDs(lane)
        if traci.vehicle.getWaitingTime(v) > 30
    )
    return passed - 2 * wait_penalty

# --- REINFORCE Training ---
def train(policy_net, optimizer, states, actions, rewards):
    discounted_rewards = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + GAMMA * cumulative
        discounted_rewards.insert(0, cumulative)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

    with tf.GradientTape() as tape:
        logits = tf.stack([policy_net(tf.expand_dims(s, axis=0))[0] for s in states])
        action_probs = tf.gather_nd(logits, [(i, a) for i, a in enumerate(actions)])
        log_probs = tf.math.log(action_probs)
        loss = -tf.reduce_sum(log_probs * discounted_rewards)

    grads = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

# --- Main Loop ---
policy_net = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

for episode in range(EPISODES):
    os.system("python generate_routes.py")  # if you have route generator
    traci.start([SUMO_BINARY, "-c", "../trafficinter.sumocfg"])

    current_phase = 0
    step = 0
    states, actions, rewards = [], [], []
    total_reward = 0

    for _ in range(MAX_PHASES_PER_EPISODE):
        state = get_state(current_phase)
        action_probs = policy_net(tf.convert_to_tensor([state], dtype=tf.float32))[0].numpy()
        action = np.random.choice(ACTION_SIZE, p=action_probs)
        duration = DURATIONS[action]

        traci.trafficlight.setPhase(TLS_ID, current_phase)
        for _ in range(duration):
            traci.simulationStep()
            step += 1

        reward = compute_reward()
        total_reward += reward

        states.append(tf.convert_to_tensor(state))
        actions.append(action)
        rewards.append(reward)

        current_phase = (current_phase + 1) % 4

    traci.close()
    print(f"Episode {episode + 1}/{EPISODES} | Total Reward: {total_reward}")
    train(policy_net, optimizer, states, actions, rewards)

# Save model
policy_net.save_weights("logs/policy_model.h5")