import numpy as np
import tensorflow as tf
import traci
import os
from generate_routes import generate_random_routes

# Constants
CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
                    '-east_0', '-east_0', '-east_1', '-east_2',
                    '-south_0', '-south_0', '-south_1',
                    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
STATE_SIZE = 14 + 1
SUMO_BINARY = "sumo-gui"
YELLOW_PHASE_OFFSET = 4
YELLOW_DURATION = 4
MAX_STEPS = 800
VEHICLES = 300

# --- Load Policy Network ---
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.output_layer(x)

policy_net = PolicyNetwork()
policy_net.build(input_shape=(None, STATE_SIZE))
policy_net.load_weights("logs/policy_model_continuous.weights.h5")
print("Model loaded.")

# --- Helper Functions ---
def get_state(current_phase):
    queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in CONTROLLED_LANES]
    return np.array(queue_lengths + [current_phase], dtype=np.float32)

def compute_throughput():
    return len(traci.simulation.getArrivedIDList())

# --- Inference Run ---
generate_random_routes("random_routes.rou.xml", num_vehicles=VEHICLES)
traci.start([SUMO_BINARY, "-c", "../trafficinter.sumocfg"])

current_phase = 0
step = 0
total_reward = 0
total_throughput = 0

while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
    state = get_state(current_phase)
    normalized_duration = policy_net(tf.convert_to_tensor([state], dtype=tf.float32))[0].numpy()[0]
    duration = int(normalized_duration * 15 + 5)

    # Yellow phase
    traci.trafficlight.setPhase(TLS_ID, current_phase + YELLOW_PHASE_OFFSET)
    for _ in range(YELLOW_DURATION):
        traci.simulationStep()
        step += 1
        if step >= MAX_STEPS: break

    if step >= MAX_STEPS: break

    # Green phase
    traci.trafficlight.setPhase(TLS_ID, current_phase)
    for _ in range(duration):
        total_throughput += compute_throughput()
        traci.simulationStep()
        step += 1
        if step >= MAX_STEPS: break

    current_phase = (current_phase + 1) % 4

traci.close()
print(f"Inference Run Complete. Total Throughput: {total_throughput}")
