"""
Deep Q-Learning with Neural Networks
====================================
This script demonstrates Deep Q-Learning (DQN) using a neural network
to solve the CartPole environment from OpenAI Gym.
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from collections import deque
import random

# ===========================================
# 1. Define the Environment
# ===========================================
env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f"State space dimensions: {n_states}")
print(f"Number of actions: {n_actions}")

# ===========================================
# 2. Build the DQN Model
# ===========================================
def build_dqn(n_states, n_actions):
    model = Sequential([
        Dense(24, activation="relu", input_shape=(n_states,)),
        Dense(24, activation="relu"),
        Dense(n_actions, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse")
    return model

# Initialize the DQN model
model = build_dqn(n_states, n_actions)
target_model = build_dqn(n_states, n_actions)
target_model.set_weights(model.get_weights())

# ===========================================
# 3. Define the Replay Buffer
# ===========================================
class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

# Initialize the replay buffer
replay_buffer = ReplayBuffer()

# ===========================================
# 4. Define the Training Loop
# ===========================================
def train_dqn(model, target_model, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Predict Q-values for current and next states
    q_values = model.predict(states)
    next_q_values = target_model.predict(next_states)

    # Update Q-values
    for i in range(batch_size):
        target = rewards[i]
        if not dones[i]:
            target += gamma * np.max(next_q_values[i])
        q_values[i][actions[i]] = target

    model.fit(states, q_values, verbose=0)

# ===========================================
# 5. Train the Agent
# ===========================================
episodes = 500
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
update_target_every = 10
reward_history = []

for episode in range(episodes):
    state = env.reset(seed=42)[0]
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state[np.newaxis])
            action = np.argmax(q_values)

        # Perform the action
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Train the model
        train_dqn(model, target_model, replay_buffer, batch_size, gamma)

    # Update the target model weights
    if episode % update_target_every == 0:
        target_model.set_weights(model.get_weights())

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_history.append(total_reward)

    # Log progress
    print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward} | Epsilon: {epsilon:.4f}")

# ===========================================
# 6. Evaluate the Trained Agent
# ===========================================
def dqn_policy(state):
    q_values = model.predict(state[np.newaxis])
    return np.argmax(q_values)

test_episodes = 20
test_rewards = []

for episode in range(test_episodes):
    state = env.reset(seed=42)[0]
    total_reward = 0
    done = False

    while not done:
        action = dqn_policy(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    test_rewards.append(total_reward)

print(f"Average reward after training: {np.mean(test_rewards):.2f}")

# ===========================================
# 7. Visualize the Training Performance
# ===========================================
plt.figure(figsize=(10, 6))
plt.plot(reward_history, label="Reward per Episode", color="blue")
plt.axhline(y=np.mean(test_rewards), color="red", linestyle="--", label="Average Test Reward")
plt.title("Deep Q-Learning Training Performance")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.show()

# ===========================================
# Summary
# ===========================================
"""
Key Concepts Covered:
1. Neural Network for Q-Value Approximation.
2. Experience Replay: Efficiently reuses past experiences for learning.
3. Epsilon-Greedy Policy: Balances exploration and exploitation.
4. Target Network: Stabilizes Q-value updates.
5. Reward Evaluation: Measures training and testing performance.

This script highlights the power of DQNs in solving environments with continuous state spaces.
"""
