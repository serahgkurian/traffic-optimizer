import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import traci
import os
import csv
from generate_routes import generate_random_routes

# Constants
GREEN_PHASES = [0, 1, 2, 3]  # Green phases from TLS config
YELLOW_PHASES = [4, 5, 6, 7]  # Corresponding yellow phases
STATE_SIZE = 18  # Simplified state: 14 queue lengths + 4 phase features
GAMMA = 0.99
LEARNING_RATE = 0.001
EPISODES = 150
MAX_STEPS = 1000
CONTROLLED_LANES = ['-north_0', '-north_0', '-north_1',
                    '-east_0', '-east_0', '-east_1', '-east_2',
                    '-south_0', '-south_0', '-south_1',
                    '-west_0', '-west_0', '-west_1', '-west_2']
TLS_ID = "J0"
VEHICLES_PER_RUN = 500
SUMO_BINARY = "sumo"
YELLOW_DURATION = 4

# Queue penalty thresholds
QUEUE_PENALTY_THRESHOLD = 5  # Start penalizing when queue > 5 vehicles
MAX_QUEUE_PENALTY = 3.0      # Maximum penalty per lane for very long queues

# Simple Policy Network
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(32, activation='relu')
        self.duration_mean = tf.keras.layers.Dense(1, activation='sigmoid')
        self.duration_std = tf.keras.layers.Dense(1, activation='softplus')
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mean = self.duration_mean(x)
        std = self.duration_std(x) + 1e-6
        return mean, std

# Simple Baseline Network
class BaselineNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')
        self.fc2 = tf.keras.layers.Dense(16, activation='relu')
        self.value_head = tf.keras.layers.Dense(1)
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.value_head(x)

def get_simple_state(current_phase_index):
    """Simplified state with just queue lengths and phase info"""
    try:
        # Get queue lengths for all controlled lanes
        queue_lengths = []
        for lane in CONTROLLED_LANES:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                queue_lengths.append(queue)
            except:
                queue_lengths.append(0)
        
        # Simple phase features
        phase_features = [
            current_phase_index / 3.0,  # Normalized phase index
            np.sin(2 * np.pi * current_phase_index / 4),  # Cyclic encoding
            np.cos(2 * np.pi * current_phase_index / 4),
            1.0  # Bias term
        ]
        
        state = np.array(queue_lengths + phase_features, dtype=np.float32)
        return np.clip(state, 0, 20)  # Simple clipping
        
    except Exception as e:
        print(f"Error getting state: {e}")
        return np.zeros(STATE_SIZE, dtype=np.float32)

def compute_enhanced_reward(overall_throughput_rate):
    """Enhanced reward based on overall throughput rate with queue penalties"""
    try:
        # Reward based on overall throughput rate (vehicles per minute)
        throughput_reward = overall_throughput_rate * 15.0  # Higher multiplier for throughput rate
        
        # Penalty for waiting vehicles (existing)
        wait_penalty = 0
        for lane in CONTROLLED_LANES:
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                for veh_id in veh_ids:
                    try:
                        wait_time = traci.vehicle.getWaitingTime(veh_id)
                        if wait_time > 10:  # Only penalize long waits
                            wait_penalty += wait_time * 0.1
                    except:
                        continue
            except:
                continue
        
        # Queue length penalty
        queue_penalty = 0
        total_queue_length = 0
        for lane in CONTROLLED_LANES:
            try:
                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                total_queue_length += queue_length
                
                # Apply exponential penalty for long queues
                if queue_length > QUEUE_PENALTY_THRESHOLD:
                    # Exponential penalty: grows quickly for very long queues
                    excess_queue = queue_length - QUEUE_PENALTY_THRESHOLD
                    lane_penalty = min(MAX_QUEUE_PENALTY, 0.5 * (excess_queue ** 1.5))
                    queue_penalty += lane_penalty
                    
            except:
                continue
        
        # Additional penalty for extremely long total queues across all lanes
        if total_queue_length > 20:  # If total queue across all lanes > 20
            queue_penalty += (total_queue_length - 20) * 0.2
        
        total_reward = throughput_reward - wait_penalty - queue_penalty
        
        return total_reward, throughput_reward, wait_penalty, queue_penalty, total_queue_length
        
    except Exception as e:
        print(f"Error computing reward: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0

def get_avg_wait_time():
    """Get average waiting time across all lanes"""
    total_wait = 0
    total_vehicles = 0
    
    for lane in CONTROLLED_LANES:
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            for v in veh_ids:
                try:
                    total_wait += traci.vehicle.getWaitingTime(v)
                    total_vehicles += 1
                except:
                    continue
        except:
            continue
    
    return total_wait / total_vehicles if total_vehicles > 0 else 0

def get_avg_queue_length():
    """Get average queue length across all lanes"""
    total_queue = 0
    lane_count = 0
    
    for lane in CONTROLLED_LANES:
        try:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            total_queue += queue
            lane_count += 1
        except:
            continue
    
    return total_queue / lane_count if lane_count > 0 else 0

def train_networks(policy_net, baseline_net, policy_optimizer, baseline_optimizer,
                  states, actions, rewards):
    """Simple training function"""
    try:
        if len(states) == 0:
            return 0.0
            
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Get states as tensor
        states_tensor = tf.stack(states)
        
        # Train baseline network
        with tf.GradientTape() as baseline_tape:
            predicted_values = tf.squeeze(baseline_net(states_tensor))
            baseline_loss = tf.reduce_mean(tf.square(returns - predicted_values))
        
        baseline_grads = baseline_tape.gradient(baseline_loss, baseline_net.trainable_variables)
        baseline_optimizer.apply_gradients(zip(baseline_grads, baseline_net.trainable_variables))
        
        # Compute advantages
        baselines = tf.squeeze(baseline_net(states_tensor))
        advantages = returns - baselines
        
        # Normalize advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        # Train policy network
        with tf.GradientTape() as policy_tape:
            means, stds = zip(*[policy_net(tf.expand_dims(s, axis=0)) for s in states])
            means = tf.stack([m[0][0] for m in means])
            stds = tf.stack([s[0][0] for s in stds])
            
            # Create distribution
            dist = tfp.distributions.Normal(means, stds)
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
            log_probs = dist.log_prob(actions_tensor)
            
            # Policy loss
            policy_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
        
        policy_grads = policy_tape.gradient(policy_loss, policy_net.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_grads, policy_net.trainable_variables))
        
        return float(policy_loss)
        
    except Exception as e:
        print(f"Error in training: {e}")
        return 0.0

def main():
    # Initialize networks
    policy_net = PolicyNetwork()
    baseline_net = BaselineNetwork()
    
    # Initialize optimizers
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # CSV logging with enhanced metrics
    with open("logs/episode_metrics_enhanced.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward", "Throughput", "AvgWaitTime", "AvgQueueLength", 
                        "ThroughputReward", "EfficiencyReward", "WaitPenalty", "QueuePenalty",
                        "PolicyLoss", "Entropy", "AvgPhaseDuration"])

    print("Starting enhanced REINFORCE training with overall throughput rewards...")
    print(f"Reward based on throughput rate (vehicles per minute)")
    print(f"Queue penalty threshold: {QUEUE_PENALTY_THRESHOLD} vehicles")
    print(f"Maximum queue penalty per lane: {MAX_QUEUE_PENALTY}")
    
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
        current_phase_index = 0  # Index into GREEN_PHASES array
        step = 0
        states, durations_taken, episode_rewards = [], [], []
        total_reward = 0
        total_throughput = 0
        total_throughput_reward = 0
        total_wait_penalty = 0
        total_queue_penalty = 0
        wait_times = []
        queue_lengths = []
        episode_start_time = step
        throughput_history = []  # Track throughput over time for rate calculation
        
        try:
            while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
                # Get current green phase
                current_green_phase = GREEN_PHASES[current_phase_index]
                current_yellow_phase = YELLOW_PHASES[current_phase_index]
                
                # Get state
                state = get_simple_state(current_phase_index)
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                
                # Get duration from policy network
                mean, std = policy_net(state_tensor)
                
                # Sample duration
                dist = tfp.distributions.Normal(mean[0][0], std[0][0])
                normalized_duration = tf.clip_by_value(dist.sample(), 0.0, 1.0)
                duration = int(normalized_duration * 35 + 5)  # 5-40 seconds
                
                # Set phase duration to be very long to prevent SUMO from switching automatically
                traci.trafficlight.setPhaseDuration(TLS_ID, 10000)
                
                # Apply green phase
                traci.trafficlight.setPhase(TLS_ID, current_green_phase)
                vehicles_passed_this_phase = 0
                
                # Stay in green phase for the determined duration
                for phase_step in range(duration):
                    if step >= MAX_STEPS:
                        break
                    
                    arrived_this_step = traci.simulation.getArrivedIDList()
                    vehicles_passed_this_phase += len(arrived_this_step)
                    
                    traci.simulationStep()
                    step += 1
                
                if step >= MAX_STEPS:
                    break

                # Apply corresponding yellow phase
                traci.trafficlight.setPhase(TLS_ID, current_yellow_phase)
                traci.trafficlight.setPhaseDuration(TLS_ID, 10000)  # Prevent auto-switching
                
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

                # Calculate overall throughput rate (vehicles per minute)
                time_elapsed = step - episode_start_time
                if time_elapsed > 0:
                    current_throughput_rate = (total_throughput / time_elapsed) * 60  # vehicles per minute
                else:
                    current_throughput_rate = 0
                
                throughput_history.append(current_throughput_rate)

                # Calculate enhanced reward with overall throughput rate
                phase_reward, throughput_reward, wait_penalty, queue_penalty, queue_length = compute_enhanced_reward(current_throughput_rate)
                total_reward += phase_reward
                total_throughput_reward += throughput_reward
                total_wait_penalty += wait_penalty
                total_queue_penalty += queue_penalty

                # Collect metrics
                avg_wait = get_avg_wait_time()
                avg_queue = get_avg_queue_length()
                wait_times.append(avg_wait)
                queue_lengths.append(avg_queue)

                # Store for training
                normalized_duration_for_training = float(normalized_duration)
                states.append(tf.convert_to_tensor(state))
                durations_taken.append(normalized_duration_for_training)
                episode_rewards.append(phase_reward)

                # Move to next phase (cycle through 0, 1, 2, 3)
                current_phase_index = (current_phase_index + 1) % 4

        except Exception as e:
            print(f"Error during episode: {e}")
        finally:
            try:
                traci.close()
            except:
                pass

        # Calculate episode metrics
        avg_wait = np.mean(wait_times) if wait_times else 0
        avg_queue = np.mean(queue_lengths) if queue_lengths else 0
        avg_duration = np.mean([d * 35 + 5 for d in durations_taken]) if durations_taken else 0
        final_throughput_rate = np.mean(throughput_history) if throughput_history else 0

        # Train the networks
        policy_loss = 0.0
        if len(states) > 0:
            policy_loss = train_networks(
                policy_net, baseline_net, policy_optimizer, baseline_optimizer,
                states, durations_taken, episode_rewards)

        # Print enhanced episode summary
        print(f"Total Reward: {total_reward:.2f} | Throughput: {total_throughput} | Throughput Rate: {final_throughput_rate:.2f} veh/min")
        print(f"AvgWait: {avg_wait:.2f} | AvgQueue: {avg_queue:.2f}")
        print(f"Rewards - Throughput: {total_throughput_reward:.1f}, Wait Penalty: {total_wait_penalty:.1f}, Queue Penalty: {total_queue_penalty:.1f}")
        print(f"Phases completed: {len(states)} | Avg phase duration: {avg_duration:.1f}s | "
              f"Policy Loss: {policy_loss:.3f}")

        # Save metrics to CSV with enhanced data
        with open("logs/episode_metrics_enhanced.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1, total_reward, total_throughput, avg_wait, avg_queue,
                total_throughput_reward, 0, total_wait_penalty, total_queue_penalty,
                policy_loss, 0, avg_duration
            ])

        # Save model periodically
        if (episode + 1) % 50 == 0:
            policy_net.save_weights(f"logs/policy_model_ep{episode+1}.weights.h5")
            baseline_net.save_weights(f"logs/baseline_model_ep{episode+1}.weights.h5")

    # Save final models
    policy_net.save_weights("logs/policy_model_final.weights.h5")
    baseline_net.save_weights("logs/baseline_model_final.weights.h5")
    
    print("\nTraining completed!")
    print("Enhanced version features:")
    print("- Reward based on overall throughput rate (vehicles/minute)")
    print("- Queue length penalty (exponential for long queues)")
    print("- Detailed reward breakdown tracking")
    print("- Enhanced metrics logging")
    print("- Penalty threshold:", QUEUE_PENALTY_THRESHOLD, "vehicles")
    print("- Maximum penalty per lane:", MAX_QUEUE_PENALTY)
    print("- Additional penalty for total queue > 20 vehicles")
    print("- Throughput rate multiplier: 15.0 (higher reward for sustained flow)")

if __name__ == "__main__":
    main()