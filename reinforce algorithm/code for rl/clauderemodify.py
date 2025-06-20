import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import traci
import os
import random
import csv
import matplotlib.pyplot as plt
from collections import deque
from generate_routes import generate_random_routes

# Constants
PHASES = [0, 1, 2, 3]
STATE_SIZE = 74  # Enhanced state size (14 lanes * 5 features + 4 phase features)
GAMMA = 0.99
INITIAL_LEARNING_RATE = 0.001
EPISODES = 150
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

# Training hyperparameters
GRADIENT_CLIP = 1.0
ENTROPY_COEFF = 0.01
BATCH_SIZE = 32

# Duration range (adaptive)
def get_duration_range(episode):
    if episode < 50:
        return 10, 20    # Conservative range initially
    elif episode < 150:
        return 8, 30    # Expand range
    else:
        return 4, 40    # Full range after learning basics

# Adaptive reward weights
class AdaptiveRewards:
    def __init__(self):
        self.episode = 0
        self.throughput_history = deque(maxlen=50)
        self.wait_time_history = deque(maxlen=50)
        
    def update_episode(self, episode, throughput, avg_wait):
        self.episode = episode
        self.throughput_history.append(throughput)
        self.wait_time_history.append(avg_wait)
        
    def get_weights(self):
        # Adaptive weights based on performance
        base_throughput_weight = 15.0
        base_wait_penalty = 1.0
        
        # If throughput is consistently low, increase its weight
        if len(self.throughput_history) > 10:
            recent_throughput = np.mean(list(self.throughput_history)[-10:])
            if recent_throughput < 400:  # Adjust threshold based on your setup
                throughput_multiplier = 1.5
            else:
                throughput_multiplier = 1.0
        else:
            throughput_multiplier = 1.0
            
        return {
            'throughput': base_throughput_weight * throughput_multiplier,
            'wait_penalty': base_wait_penalty,
            'efficiency': 3.0,
            'balance': 2.0
        }

# --- Enhanced Policy Network ---
class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size=74, hidden_sizes=[128, 256, 128], dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Deeper network with batch normalization
        self.fc1 = tf.keras.layers.Dense(hidden_sizes[0], activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        self.fc2 = tf.keras.layers.Dense(hidden_sizes[1], activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        self.fc3 = tf.keras.layers.Dense(hidden_sizes[2], activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        
        # Separate heads for mean and std of duration distribution
        self.duration_mean = tf.keras.layers.Dense(1, activation='sigmoid')
        self.duration_std = tf.keras.layers.Dense(1, activation='softplus')
        
    def call(self, state, training=False):
        x = self.dropout1(self.bn1(self.fc1(state)), training=training)
        x = self.dropout2(self.bn2(self.fc2(x)), training=training)
        x = self.dropout3(self.bn3(self.fc3(x)), training=training)
        
        mean = self.duration_mean(x)
        std = self.duration_std(x) + 1e-6  # Ensure positive std
        
        return mean, std

# --- Baseline Network for Variance Reduction ---
class BaselineNetwork(tf.keras.Model):
    def __init__(self, state_size=74, hidden_sizes=[64, 128, 64]):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_sizes[0], activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_sizes[1], activation='relu')
        self.fc3 = tf.keras.layers.Dense(hidden_sizes[2], activation='relu')
        self.value_head = tf.keras.layers.Dense(1)
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.value_head(x)

# --- Training Monitor ---
class TrainingMonitor:
    def __init__(self):
        self.metrics = {
            'episode_rewards': [],
            'throughput': [],
            'wait_times': [],
            'phase_durations': [],
            'policy_loss': [],
            'entropy': []
        }
        
    def log_episode(self, episode, metrics_dict):
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Plot every 50 episodes
        if episode % 50 == 0 and episode > 0:
            self.plot_metrics(episode)
    
    def plot_metrics(self, episode):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot each metric
        if self.metrics['episode_rewards']:
            axes[0,0].plot(self.metrics['episode_rewards'])
            axes[0,0].set_title('Episode Rewards')
        
        if self.metrics['throughput']:
            axes[0,1].plot(self.metrics['throughput'])
            axes[0,1].set_title('Throughput')
        
        if self.metrics['wait_times']:
            axes[0,2].plot(self.metrics['wait_times'])
            axes[0,2].set_title('Average Wait Time')
        
        if self.metrics['phase_durations']:
            axes[1,0].plot(self.metrics['phase_durations'])
            axes[1,0].set_title('Average Phase Duration')
        
        if self.metrics['policy_loss']:
            axes[1,1].plot(self.metrics['policy_loss'])
            axes[1,1].set_title('Policy Loss')
        
        if self.metrics['entropy']:
            axes[1,2].plot(self.metrics['entropy'])
            axes[1,2].set_title('Policy Entropy')
        
        plt.tight_layout()
        plt.savefig(f'logs/training_progress_ep{episode}.png')
        plt.close()

# --- Helper Functions ---
def get_enhanced_state(current_phase, phase_duration=0):
    """Enhanced state representation with more traffic features"""
    try:
        # Current features - queue lengths
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in CONTROLLED_LANES]
        
        # New features
        vehicle_speeds = []
        waiting_times = []
        approach_vehicles = []  # Vehicles approaching intersection
        
        for lane in CONTROLLED_LANES:
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                if veh_ids:
                    speeds = []
                    waits = []
                    approaching = 0
                    
                    for v in veh_ids:
                        try:
                            speed = traci.vehicle.getSpeed(v)
                            wait = traci.vehicle.getWaitingTime(v)
                            speeds.append(speed)
                            waits.append(wait)
                            
                            # Count approaching vehicles (simple heuristic)
                            if speed > 0.1:  # Moving vehicles
                                approaching += 1
                                
                        except traci.exceptions.TraCIException:
                            continue
                    
                    vehicle_speeds.append(np.mean(speeds) if speeds else 0)
                    waiting_times.append(np.mean(waits) if waits else 0)
                    approach_vehicles.append(approaching)
                else:
                    vehicle_speeds.append(0)
                    waiting_times.append(0)
                    approach_vehicles.append(0)
                    
            except traci.exceptions.TraCIException:
                vehicle_speeds.append(0)
                waiting_times.append(0)
                approach_vehicles.append(0)
        
        # Traffic flow rates (vehicles per step)
        flow_rates = []
        for lane in CONTROLLED_LANES:
            try:
                flow_rates.append(traci.lane.getLastStepVehicleNumber(lane))
            except traci.exceptions.TraCIException:
                flow_rates.append(0)
        
        # Phase timing information
        phase_features = [
            current_phase / 3.0,  # Normalized phase
            phase_duration / 50.0,  # Normalized duration
            np.sin(2 * np.pi * current_phase / 4),  # Cyclic encoding
            np.cos(2 * np.pi * current_phase / 4)
        ]
        
        # Combine all features
        state = np.array(queue_lengths + vehicle_speeds + waiting_times + 
                        approach_vehicles + flow_rates + phase_features, dtype=np.float32)
        
        # Simple normalization to prevent extreme values
        state = np.clip(state, -10, 10)
        
        return state
        
    except Exception as e:
        print(f"Error in get_enhanced_state: {e}")
        # Return a default state if there's an error
        return np.zeros(STATE_SIZE, dtype=np.float32)

def get_green_lanes(current_phase):
    """Get the lanes that have green light for the current phase"""
    try:
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
    except:
        return []

def get_direction_waiting_times():
    """Get maximum waiting time per direction"""
    direction_waits = {'-north': 0, '-east': 0, '-south': 0, '-west': 0}
    
    for lane in CONTROLLED_LANES:
        direction = lane.split('_')[0]
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            max_wait = 0
            for veh_id in veh_ids:
                try:
                    wait_time = traci.vehicle.getWaitingTime(veh_id)
                    max_wait = max(max_wait, wait_time)
                except:
                    continue
            direction_waits[direction] = max(direction_waits[direction], max_wait)
        except:
            continue
    
    return direction_waits

def compute_balanced_reward(current_phase, phase_duration, vehicles_passed, weights):
    """Simplified but more effective reward function"""
    try:
        # Primary: Throughput
        throughput_reward = max(0, vehicles_passed) * weights['throughput']
        
        # Secondary: Waiting time penalty (exponential to heavily penalize long waits)
        total_wait_penalty = 0
        for lane in CONTROLLED_LANES:
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                for veh_id in veh_ids:
                    try:
                        wait_time = traci.vehicle.getWaitingTime(veh_id)
                        # Exponential penalty for long waits
                        if wait_time > 5:  # Only penalize significant waits
                            total_wait_penalty += np.exp(wait_time / 30.0) - 1
                    except:
                        continue
            except:
                continue
        
        wait_penalty = total_wait_penalty * weights['wait_penalty']
        
        # Efficiency: Reward for keeping traffic moving in green lanes
        green_lanes = get_green_lanes(current_phase)
        efficiency_reward = 0
        for lane in green_lanes:
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                moving_vehicles = sum(1 for v in veh_ids if traci.vehicle.getSpeed(v) > 0.5)
                efficiency_reward += moving_vehicles * weights['efficiency']
            except:
                continue
        
        # Balance: Prevent starvation
        direction_waits = get_direction_waiting_times()
        if direction_waits:
            max_wait_diff = max(direction_waits.values()) - min(direction_waits.values())
            balance_penalty = max_wait_diff * weights['balance']
        else:
            balance_penalty = 0
        
        total_reward = throughput_reward + efficiency_reward - wait_penalty - balance_penalty
        
        return total_reward, {
            'throughput': throughput_reward,
            'efficiency': efficiency_reward,
            'wait_penalty': -wait_penalty,
            'balance_penalty': -balance_penalty
        }
        
    except Exception as e:
        print(f"Error in compute_balanced_reward: {e}")
        return 0.0, {'throughput': 0, 'efficiency': 0, 'wait_penalty': 0, 'balance_penalty': 0}

def get_avg_wait_and_queue():
    """Get average waiting time and queue length across all controlled lanes"""
    total_wait = 0
    total_queue = 0
    total_vehicles = 0
    
    for lane in CONTROLLED_LANES:
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            for v in veh_ids:
                try:
                    total_wait += traci.vehicle.getWaitingTime(v)
                    if traci.vehicle.getSpeed(v) < 0.1:
                        total_queue += 1
                    total_vehicles += 1
                except:
                    continue
        except:
            continue
    
    avg_wait = total_wait / total_vehicles if total_vehicles > 0 else 0
    avg_queue = total_queue / len(CONTROLLED_LANES) if CONTROLLED_LANES else 0
    
    return avg_wait, avg_queue

def get_learning_rate(episode, initial_lr=0.001):
    """Learning rate scheduling"""
    if episode < 100:
        return initial_lr
    elif episode < 200:
        return initial_lr * 0.5
    else:
        return initial_lr * 0.1

# --- Enhanced Training Function ---
def train_with_baseline(policy_net, baseline_net, policy_optimizer, baseline_optimizer, 
                       states, actions, rewards):
    """Training with baseline to reduce variance"""
    try:
        if len(states) == 0:
            return 0.0, 0.0
            
        rewards_array = np.array(rewards)
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards_array):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Compute baseline values
        states_tensor = tf.stack(states)
        baselines = tf.squeeze(baseline_net(states_tensor))
        
        # Compute advantages
        advantages = returns - baselines
        
        # Normalize advantages
        if tf.reduce_std(advantages) > 1e-8:
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.reduce_std(advantages) + 1e-8)
        
        # Train baseline network
        with tf.GradientTape() as baseline_tape:
            predicted_values = tf.squeeze(baseline_net(states_tensor))
            baseline_loss = tf.reduce_mean(tf.square(returns - predicted_values))
        
        baseline_grads = baseline_tape.gradient(baseline_loss, baseline_net.trainable_variables)
        baseline_grads = [tf.clip_by_norm(g, GRADIENT_CLIP) for g in baseline_grads]
        baseline_optimizer.apply_gradients(zip(baseline_grads, baseline_net.trainable_variables))
        
        # Train policy network
        with tf.GradientTape() as policy_tape:
            # Get policy outputs
            means, stds = zip(*[policy_net(tf.expand_dims(s, axis=0), training=True) for s in states])
            means = tf.stack([m[0][0] for m in means])
            stds = tf.stack([s[0][0] for s in stds])
            
            # Create distribution and compute log probabilities
            dist = tfp.distributions.Normal(means, stds)
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
            log_probs = dist.log_prob(actions_tensor)
            
            # Policy gradient loss
            policy_loss = -tf.reduce_sum(log_probs * tf.stop_gradient(advantages))
            
            # Entropy bonus for exploration
            entropy_loss = -tf.reduce_sum(dist.entropy()) * ENTROPY_COEFF
            
            total_loss = policy_loss + entropy_loss
        
        # Apply policy gradients
        policy_grads = policy_tape.gradient(total_loss, policy_net.trainable_variables)
        policy_grads = [tf.clip_by_norm(g, GRADIENT_CLIP) for g in policy_grads if g is not None]
        policy_optimizer.apply_gradients(zip(policy_grads, policy_net.trainable_variables))
        
        return float(total_loss), float(tf.reduce_mean(dist.entropy()))
        
    except Exception as e:
        print(f"Error in training: {e}")
        return 0.0, 0.0

# --- Main Training Loop ---
def main():
    # Initialize networks
    policy_net = PolicyNetwork(state_size=STATE_SIZE)
    baseline_net = BaselineNetwork(state_size=STATE_SIZE)
    
    # Initialize optimizers
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    
    # Initialize adaptive rewards and monitor
    adaptive_rewards = AdaptiveRewards()
    monitor = TrainingMonitor()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # CSV logging
    with open("logs/episode_metrics_enhanced.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward", "Throughput", "AvgWaitTime", "AvgQueueLength", 
                        "ThroughputReward", "EfficiencyReward", "WaitPenalty", "BalancePenalty",
                        "PolicyLoss", "Entropy", "AvgPhaseDuration"])

    print("Starting enhanced REINFORCE training...")
    
    for episode in range(EPISODES):
        print(f"\n--- Episode {episode + 1}/{EPISODES} ---")
        
        # Generate routes
        route_file = "../code for rl/random_routes.rou.xml"
        generate_random_routes(route_file, num_vehicles=VEHICLES_PER_RUN)

        # Show GUI on last episode
        sumo_binary = "sumo-gui" if episode == EPISODES - 1 else SUMO_BINARY
        
        try:
            traci.start([sumo_binary, "-c", "../trafficinter.sumocfg"])
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            continue

        # Episode variables
        current_phase = 0
        step = 0
        states, durations_taken, episode_rewards = [], [], []
        total_reward = 0
        total_throughput = 0
        wait_time_accum = []
        queue_len_accum = []
        
        # Get adaptive reward weights
        adaptive_rewards.update_episode(episode, 0, 0)  # Will update with actual values
        weights = adaptive_rewards.get_weights()
        
        # Get duration range for this episode
        min_dur, max_dur = get_duration_range(episode)
        
        try:
            while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
                # Get enhanced state
                state = get_enhanced_state(current_phase, 0)
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                
                # Get duration from policy network
                mean, std = policy_net(state_tensor, training=False)
                
                # Sample duration from the distribution
                dist = tfp.distributions.Normal(mean[0][0], std[0][0])
                normalized_duration = dist.sample()
                
                # Clip and scale to duration range
                normalized_duration = tf.clip_by_value(normalized_duration, 0.0, 1.0)
                duration = int(normalized_duration * (max_dur - min_dur) + min_dur)
                
                # Apply green phase and track vehicles passed
                traci.trafficlight.setPhase(TLS_ID, current_phase)
                vehicles_passed_this_phase = 0
                
                for phase_step in range(duration):
                    if step >= MAX_STEPS:
                        break
                        
                    # Count vehicles that complete their journey
                    arrived_this_step = traci.simulation.getArrivedIDList()
                    vehicles_passed_this_phase += len(arrived_this_step)
                    
                    traci.simulationStep()
                    step += 1
                
                if step >= MAX_STEPS:
                    break

                # Apply yellow phase
                yellow_phase = current_phase + YELLOW_PHASE_OFFSET
                traci.trafficlight.setPhase(TLS_ID, yellow_phase)
                for _ in range(YELLOW_DURATION):
                    if step >= MAX_STEPS:
                        break
                        
                    arrived_this_step = traci.simulation.getArrivedIDList()
                    vehicles_passed_this_phase += len(arrived_this_step)
                    
                    traci.simulationStep()
                    step += 1

                if step >= MAX_STEPS:
                    break

                total_throughput += vehicles_passed_this_phase

                # Calculate reward
                phase_reward, reward_components = compute_balanced_reward(
                    current_phase, duration, vehicles_passed_this_phase, weights)
                total_reward += phase_reward

                # Collect metrics
                avg_wait, avg_queue = get_avg_wait_and_queue()
                wait_time_accum.append(avg_wait)
                queue_len_accum.append(avg_queue)

                # Store for training (normalize duration for training)
                normalized_duration_for_training = (duration - min_dur) / (max_dur - min_dur)
                states.append(tf.convert_to_tensor(state))
                durations_taken.append(normalized_duration_for_training)
                episode_rewards.append(phase_reward)

                current_phase = (current_phase + 1) % 4

        except Exception as e:
            print(f"Error during episode: {e}")
        finally:
            try:
                traci.close()
            except:
                pass

        # Calculate episode metrics
        avg_wait = np.mean(wait_time_accum) if wait_time_accum else 0
        avg_queue = np.mean(queue_len_accum) if queue_len_accum else 0
        avg_duration = np.mean([d * (max_dur - min_dur) + min_dur for d in durations_taken]) if durations_taken else 0

        # Update adaptive rewards with actual performance
        adaptive_rewards.update_episode(episode, total_throughput, avg_wait)

        # Train the networks
        policy_loss, entropy = 0.0, 0.0
        if len(states) > 0:
            # Update learning rates
            current_lr = get_learning_rate(episode, INITIAL_LEARNING_RATE)
            policy_optimizer.learning_rate = current_lr
            baseline_optimizer.learning_rate = current_lr
            
            policy_loss, entropy = train_with_baseline(
                policy_net, baseline_net, policy_optimizer, baseline_optimizer,
                states, durations_taken, episode_rewards)

        # Print episode summary
        print(f"Total Reward: {total_reward:.2f} | Throughput: {total_throughput} | "
              f"AvgWait: {avg_wait:.2f} | AvgQueue: {avg_queue:.2f}")
        print(f"Phases completed: {len(states)} | Avg phase duration: {avg_duration:.1f}s | "
              f"Policy Loss: {policy_loss:.3f} | Entropy: {entropy:.3f}")

        # Log metrics
        metrics_dict = {
            'episode_rewards': total_reward,
            'throughput': total_throughput,
            'wait_times': avg_wait,
            'phase_durations': avg_duration,
            'policy_loss': policy_loss,
            'entropy': entropy
        }
        monitor.log_episode(episode, metrics_dict)

        # Save detailed metrics to CSV
        with open("logs/episode_metrics_enhanced.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1, total_reward, total_throughput, avg_wait, avg_queue,
                0, 0, 0, 0,  # Placeholder for reward components
                policy_loss, entropy, avg_duration
            ])

        # Save model periodically
        if (episode + 1) % 50 == 0:
            policy_net.save_weights(f"logs/policy_model_ep{episode+1}.weights.h5")
            baseline_net.save_weights(f"logs/baseline_model_ep{episode+1}.weights.h5")

    # Save final models
    policy_net.save_weights("logs/policy_model_final.weights.h5")
    baseline_net.save_weights("logs/baseline_model_final.weights.h5")
    
    print("\nTraining completed!")
    print("Enhanced features implemented:")
    print("- Deeper policy network with batch normalization")
    print("- Enhanced state representation (60 features)")
    print("- Baseline network for variance reduction")
    print("- Adaptive reward weights")
    print("- Learning rate scheduling")
    print("- Gradient clipping and entropy regularization")
    print("- Comprehensive monitoring and plotting")

if __name__ == "__main__":
    main()