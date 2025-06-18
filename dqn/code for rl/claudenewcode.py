import numpy as np
import traci
from collections import deque, namedtuple
import random
import tensorflow as tf
from tensorflow import keras
import csv
import os
import matplotlib.pyplot as plt
from generate_routes import generate_random_routes

# Configuration
class Config:
    # Environment
    CONTROLLED_LANES = ['-north_0', '-north_1', '-east_0', '-east_1', '-east_2', 
                       '-south_0', '-south_1', '-west_0', '-west_1', '-west_2']
    TLS_ID = "J0"
    ACTION_SIZE = 4  # Number of traffic light phases
    SUMO_BINARY = "sumo-gui"
    VEHICLES_PER_RUN = 300
    MAX_STEPS = 500
    EPISODES = 300
    
    # DQN Parameters
    STATE_SIZE = len(CONTROLLED_LANES) * 3 + ACTION_SIZE + 2  # Enhanced state space
    GAMMA = 0.95
    LEARNING_RATE = 0.0005
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 200
    
    # Training Parameters
    BATCH_SIZE = 64
    MEMORY_SIZE = 50000
    TARGET_UPDATE_FREQ = 1000
    TRAIN_FREQ = 4
    MIN_REPLAY_SIZE = 1000
    
    # Reward Parameters
    THROUGHPUT_WEIGHT = 3.0
    WAITING_WEIGHT = 1.0
    QUEUE_WEIGHT = 0.5
    PHASE_CHANGE_PENALTY = 1.0
    LONG_WAIT_THRESHOLD = 45.0
    LONG_WAIT_PENALTY = 3.0

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None, None, None
        
        N = len(self.buffer)
        priorities = self.priorities[:N]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(N, batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class AdvancedDQNAgent:
    """Advanced DQN Agent with Double DQN, Dueling Architecture, and Prioritized Replay"""
    
    def __init__(self, config):
        self.config = config
        self.memory = PrioritizedReplayBuffer(config.MEMORY_SIZE)
        self.main_model = self._build_dueling_model()
        self.target_model = self._build_dueling_model()
        self.update_target_model()
        
        self.step_count = 0
        self.epsilon = config.EPSILON_START
        
    def _build_dueling_model(self):
        """Build Dueling DQN architecture"""
        input_layer = keras.layers.Input(shape=(self.config.STATE_SIZE,))
        
        # Shared layers
        x = keras.layers.Dense(256, activation='relu')(input_layer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        # Value stream
        value_stream = keras.layers.Dense(128, activation='relu')(x)
        value_stream = keras.layers.Dense(1, activation='linear', name='value')(value_stream)
        
        # Advantage stream
        advantage_stream = keras.layers.Dense(128, activation='relu')(x)
        advantage_stream = keras.layers.Dense(self.config.ACTION_SIZE, activation='linear', 
                                            name='advantage')(advantage_stream)
        
        # Combine value and advantage streams
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        advantage_mean = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage_stream)
        advantage_normalized = keras.layers.Subtract()([advantage_stream, advantage_mean])
        q_values = keras.layers.Add()([value_stream, advantage_normalized])
        
        model = keras.Model(inputs=input_layer, outputs=q_values)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                     loss='huber')
        return model

    def update_target_model(self):
        """Soft update of target network"""
        tau = 0.005  # Soft update parameter
        main_weights = self.main_model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(main_weights)):
            target_weights[i] = tau * main_weights[i] + (1 - tau) * target_weights[i]
        
        self.target_model.set_weights(target_weights)

    def get_epsilon(self):
        """Exponential epsilon decay"""
        progress = min(self.step_count / self.config.EPSILON_DECAY_STEPS, 1.0)
        return self.config.EPSILON_END + (self.config.EPSILON_START - self.config.EPSILON_END) * \
               np.exp(-progress * 5)

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        self.epsilon = self.get_epsilon()
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.config.ACTION_SIZE)
        
        q_values = self.main_model.predict(state[np.newaxis, :], verbose=0)[0]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in prioritized replay buffer"""
        self.memory.add(state, action, reward, next_state, done)

    def train(self):
        """Train the agent using prioritized experience replay"""
        if len(self.memory.buffer) < self.config.MIN_REPLAY_SIZE:
            return None
        
        # Sample from prioritized replay buffer
        experiences, indices, weights = self.memory.sample(self.config.BATCH_SIZE)
        if experiences is None:
            return None
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # Double DQN: Use main network to select actions, target network to evaluate
        next_q_values_main = self.main_model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_values_main, axis=1)
        
        next_q_values_target = self.target_model.predict(next_states, verbose=0)
        next_q_values = next_q_values_target[np.arange(len(next_actions)), next_actions]
        
        # Compute target Q-values
        target_q_values = rewards + (self.config.GAMMA * next_q_values * (1 - dones))
        
        # Get current Q-values
        current_q_values = self.main_model.predict(states, verbose=0)
        target_q_values_full = current_q_values.copy()
        target_q_values_full[np.arange(len(actions)), actions] = target_q_values
        
        # Train with importance sampling weights
        sample_weights = weights if weights is not None else np.ones(len(experiences))
        history = self.main_model.fit(states, target_q_values_full, 
                                    sample_weight=sample_weights,
                                    epochs=1, verbose=0)
        
        # Update priorities
        if indices is not None:
            td_errors = np.abs(current_q_values[np.arange(len(actions)), actions] - target_q_values)
            new_priorities = td_errors + 1e-6
            self.memory.update_priorities(indices, new_priorities)
        
        self.step_count += 1
        
        # Update target network
        if self.step_count % self.config.TARGET_UPDATE_FREQ == 0:
            self.update_target_model()
        
        return history.history['loss'][0]

class TrafficEnvironment:
    """Enhanced traffic environment with better state representation"""
    
    def __init__(self, config):
        self.config = config
        self.last_phase = -1
        self.last_phase_time = 0
        self.phase_history = deque(maxlen=10)
        
    def get_enhanced_state(self, sim_step):
        """Get enhanced state representation with more traffic information"""
        tls_id = self.config.TLS_ID
        controlled_lanes = self.config.CONTROLLED_LANES
        
        # Traffic metrics per lane
        queue_lengths = []
        vehicle_speeds = []
        occupancies = []
        
        for lane in controlled_lanes:
            # Queue length (halting vehicles)
            queue_lengths.append(traci.lane.getLastStepHaltingNumber(lane))
            
            # Average speed
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            if vehicles:
                avg_speed = np.mean([traci.vehicle.getSpeed(vid) for vid in vehicles])
                vehicle_speeds.append(avg_speed / 13.89)  # Normalize by max speed (50 km/h)
            else:
                vehicle_speeds.append(1.0)  # No vehicles = free flow
            
            # Lane occupancy
            occupancy = traci.lane.getLastStepOccupancy(lane)
            occupancies.append(occupancy / 100.0)  # Normalize percentage
        
        # Traffic light state
        current_phase = traci.trafficlight.getPhase(tls_id)
        phase_one_hot = [1 if i == current_phase else 0 for i in range(self.config.ACTION_SIZE)]
        
        # Phase duration
        if current_phase == self.last_phase:
            phase_duration = sim_step - self.last_phase_time
        else:
            phase_duration = 0
            self.last_phase_time = sim_step
            self.last_phase = current_phase
        
        # Normalize phase duration
        normalized_duration = min(phase_duration / 60.0, 1.0)
        
        # Phase change frequency (stability metric)
        self.phase_history.append(current_phase)
        phase_changes = sum(1 for i in range(1, len(self.phase_history)) 
                          if self.phase_history[i] != self.phase_history[i-1])
        phase_stability = 1.0 - (phase_changes / max(len(self.phase_history) - 1, 1))
        
        # Combine all features
        state = np.array(queue_lengths + vehicle_speeds + occupancies + 
                        phase_one_hot + [normalized_duration, phase_stability], 
                        dtype=np.float32)
        
        # Normalize queue lengths
        state[:len(queue_lengths)] = np.clip(state[:len(queue_lengths)] / 20.0, 0, 1)
        
        return state, current_phase

    def calculate_reward(self, prev_metrics, current_phase):
        """Advanced reward function balancing multiple objectives"""
        controlled_lanes = self.config.CONTROLLED_LANES
        
        # Current metrics
        current_stopped = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
        current_throughput = len(traci.simulation.getArrivedIDList())
        
        # Total waiting time
        total_waiting_time = 0
        long_wait_vehicles = 0
        
        for lane in controlled_lanes:
            for vid in traci.lane.getLastStepVehicleIDs(lane):
                wait_time = traci.vehicle.getWaitingTime(vid)
                total_waiting_time += wait_time
                if wait_time > self.config.LONG_WAIT_THRESHOLD:
                    long_wait_vehicles += 1
        
        # Average speed (traffic flow quality)
        total_vehicles = 0
        total_speed = 0
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            total_vehicles += len(vehicles)
            total_speed += sum(traci.vehicle.getSpeed(vid) for vid in vehicles)
        
        avg_speed = total_speed / max(total_vehicles, 1)
        speed_efficiency = avg_speed / 13.89  # Normalize by max speed
        
        # Reward components
        throughput_reward = self.config.THROUGHPUT_WEIGHT * (current_throughput - prev_metrics['throughput'])
        waiting_penalty = -self.config.WAITING_WEIGHT * (total_waiting_time - prev_metrics['waiting_time']) / 100.0
        queue_penalty = -self.config.QUEUE_WEIGHT * current_stopped
        speed_reward = 2.0 * speed_efficiency
        long_wait_penalty = -self.config.LONG_WAIT_PENALTY * long_wait_vehicles
        
        # Phase change penalty (encourage stability but not stagnation)
        phase_change_penalty = 0
        if current_phase != prev_metrics['phase']:
            if prev_metrics['phase_duration'] < 5:  # Discourage very short phases
                phase_change_penalty = -self.config.PHASE_CHANGE_PENALTY * 2
            else:
                phase_change_penalty = -self.config.PHASE_CHANGE_PENALTY
        
        total_reward = (throughput_reward + waiting_penalty + queue_penalty + 
                       speed_reward + long_wait_penalty + phase_change_penalty)
        
        # Update metrics
        current_metrics = {
            'stopped': current_stopped,
            'throughput': current_throughput,
            'waiting_time': total_waiting_time,
            'phase': current_phase,
            'phase_duration': prev_metrics.get('phase_duration', 0) + 1 if current_phase == prev_metrics.get('phase', -1) else 0,
            'avg_speed': avg_speed,
            'long_wait_vehicles': long_wait_vehicles
        }
        
        return total_reward, current_metrics

def train_advanced_dqn():
    """Main training function"""
    config = Config()
    agent = AdvancedDQNAgent(config)
    env = TrafficEnvironment(config)
    
    # Logging setup
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/advanced_training_metrics.csv"
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward", "AvgWaitingTime", "AvgQueueLength", 
                        "Throughput", "AvgSpeed", "Epsilon", "Loss"])
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    
    for episode in range(config.EPISODES):
        # Generate routes
        route_file = r"../code for rl/random_routes.rou.xml"
        generate_random_routes(route_file, num_vehicles=config.VEHICLES_PER_RUN)
        
        # Start SUMO
        sumo_binary = "sumo-gui" if episode == config.EPISODES - 1 else config.SUMO_BINARY
        traci.start([sumo_binary, "-c", "../trafficinter.sumocfg"])
        
        # Episode initialization
        step = 0
        total_reward = 0
        episode_loss = []
        
        # Initialize metrics
        prev_metrics = {
            'stopped': 0,
            'throughput': 0,
            'waiting_time': 0,
            'phase': -1,
            'phase_duration': 0,
            'avg_speed': 0,
            'long_wait_vehicles': 0
        }
        
        # Get initial state
        state, current_phase = env.get_enhanced_state(step)
        prev_metrics['phase'] = current_phase
        
        # Episode metrics tracking
        wait_times = []
        queue_lengths = []
        speeds = []
        
        while step < config.MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
            # Choose and execute action
            action = agent.act(state)
            traci.trafficlight.setPhase(config.TLS_ID, action)
            traci.simulationStep()
            step += 1
            
            # Get next state and reward
            next_state, current_phase = env.get_enhanced_state(step)
            reward, current_metrics = env.calculate_reward(prev_metrics, current_phase)
            
            # Check if episode is done
            done = (step >= config.MAX_STEPS or traci.simulation.getMinExpectedNumber() == 0)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if step % config.TRAIN_FREQ == 0:
                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)
            
            # Update metrics
            total_reward += reward
            wait_times.append(current_metrics['waiting_time'])
            queue_lengths.append(current_metrics['stopped'])
            speeds.append(current_metrics['avg_speed'])
            
            # Move to next state
            state = next_state
            prev_metrics = current_metrics
            
            if done:
                break
        
        traci.close()
        
        # Episode statistics
        avg_wait = np.mean(wait_times) if wait_times else 0
        avg_queue = np.mean(queue_lengths) if queue_lengths else 0
        avg_speed = np.mean(speeds) if speeds else 0
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        total_throughput = current_metrics['throughput']
        
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        
        print(f"Episode {episode+1}/{config.EPISODES} | "
              f"Reward: {total_reward:.2f} | "
              f"AvgWait: {avg_wait:.2f} | "
              f"AvgQueue: {avg_queue:.2f} | "
              f"Throughput: {total_throughput} | "
              f"AvgSpeed: {avg_speed:.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.4f}")
        
        # Log metrics
        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward, avg_wait, avg_queue, 
                            total_throughput, avg_speed, agent.epsilon, avg_loss])
        
        # Save model periodically
        if (episode + 1) % 50 == 0:
            agent.main_model.save(f"logs/advanced_dqn_episode_{episode+1}.h5")
            print(f"Model saved at episode {episode+1}")
    
    # Final model save
    agent.main_model.save("logs/advanced_dqn_final_model.h5")
    agent.target_model.save("logs/advanced_dqn_target_model.h5")
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_losses)
    
    print("Training completed!")
    return agent

def plot_training_progress(rewards, losses):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot losses
    if losses and any(l > 0 for l in losses):
        ax2.plot([l for l in losses if l > 0])
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('logs/training_progress.png')
    plt.show()

def evaluate_model(model_path, num_episodes=10):
    """Evaluate trained model"""
    config = Config()
    env = TrafficEnvironment(config)
    
    # Load model
    model = keras.models.load_model(model_path)
    
    total_rewards = []
    total_wait_times = []
    total_throughputs = []
    
    for episode in range(num_episodes):
        route_file = r"../code for rl/random_routes.rou.xml"
        generate_random_routes(route_file, num_vehicles=config.VEHICLES_PER_RUN)
        
        traci.start([config.SUMO_BINARY, "-c", "../trafficinter.sumocfg"])
        
        step = 0
        total_reward = 0
        prev_metrics = {
            'stopped': 0, 'throughput': 0, 'waiting_time': 0,
            'phase': -1, 'phase_duration': 0, 'avg_speed': 0,
            'long_wait_vehicles': 0
        }
        
        state, current_phase = env.get_enhanced_state(step)
        prev_metrics['phase'] = current_phase
        
        wait_times = []
        
        while step < config.MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
            # Choose action greedily (no exploration)
            q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
            action = np.argmax(q_values)
            
            traci.trafficlight.setPhase(config.TLS_ID, action)
            traci.simulationStep()
            step += 1
            
            next_state, current_phase = env.get_enhanced_state(step)
            reward, current_metrics = env.calculate_reward(prev_metrics, current_phase)
            
            total_reward += reward
            wait_times.append(current_metrics['waiting_time'])
            
            state = next_state
            prev_metrics = current_metrics
            
            if step >= config.MAX_STEPS or traci.simulation.getMinExpectedNumber() == 0:
                break
        
        traci.close()
        
        avg_wait = np.mean(wait_times) if wait_times else 0
        throughput = current_metrics['throughput']
        
        total_rewards.append(total_reward)
        total_wait_times.append(avg_wait)
        total_throughputs.append(throughput)
        
        print(f"Eval Episode {episode+1}: Reward={total_reward:.2f}, "
              f"AvgWait={avg_wait:.2f}, Throughput={throughput}")
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Wait Time: {np.mean(total_wait_times):.2f} ± {np.std(total_wait_times):.2f}")
    print(f"Average Throughput: {np.mean(total_throughputs):.2f} ± {np.std(total_throughputs):.2f}")

# Main execution
if __name__ == "__main__":
    # Train the advanced DQN agent
    trained_agent = train_advanced_dqn()
    
    # Optionally evaluate the trained model
    # evaluate_model("logs/advanced_dqn_final_model.h5")