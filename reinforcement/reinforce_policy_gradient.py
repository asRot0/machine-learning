"""
Policy Gradient with REINFORCE
==============================
This script implements the REINFORCE algorithm, a basic policy gradient
method, to solve the CartPole-v1 environment.
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# ===========================================
# 1. Define the Environment
# ===========================================
env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f"State space dimensions: {n_states}")
print(f"Number of actions: {n_actions}")

# ===========================================
# 2. Build the Policy Network
# ===========================================
def build_policy_network(n_states, n_actions):
    model = Sequential([
        Dense(24, activation="relu", input_shape=(n_states,)),
        Dense(24, activation="relu"),
        Dense(n_actions, activation="softmax")  # Outputs probabilities for actions
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    return model, optimizer

# Initialize the policy network
policy_network, optimizer = build_policy_network(n_states, n_actions)

# ===========================================
# 3. Define the Policy Gradient Update
# ===========================================
def compute_loss(probabilities, actions, discounted_rewards):
    indices = tf.range(len(actions))
    action_probabilities = tf.gather(probabilities, indices, axis=0, batch_dims=1)
    log_probabilities = tf.math.log(action_probabilities)
    return -tf.reduce_mean(log_probabilities * discounted_rewards)

def train_policy_network(states, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        probabilities = policy_network(states, training=True)
        loss = compute_loss(probabilities, actions, discounted_rewards)
    gradients = tape.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))
    return loss

# ===========================================
# 4. Compute Discounted Rewards
# ===========================================
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative_reward = 0.0
    for t in reversed(range(len(rewards))):
        cumulative_reward = rewards[t] + gamma * cumulative_reward
        discounted_rewards[t] = cumulative_reward
    # Normalize rewards for stability
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= (np.std(discounted_rewards) + 1e-9)
    return discounted_rewards

# ===========================================
# 5. Train the Agent
# ===========================================
episodes = 500
reward_history = []
gamma = 0.99

for episode in range(episodes):
    states = []
    actions = []
    rewards = []

    state = env.reset(seed=42)[0]
    done = False
    total_reward = 0

    while not done:
        # Get action probabilities
        state_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        action_probs = policy_network(state_tensor).numpy().flatten()

        # Sample action from probabilities
        action = np.random.choice(n_actions, p=action_probs)
        next_state, reward, done, _, _ = env.step(action)

        # Store trajectory
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        total_reward += reward

    # Convert to tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    discounted_rewards = compute_discounted_rewards(rewards, gamma)

    # Train the policy network
    loss = train_policy_network(states, actions, discounted_rewards)

    reward_history.append(total_reward)
    print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward:.2f} | Loss: {loss.numpy():.4f}")

# ===========================================
# 6. Evaluate the Trained Agent
# ===========================================
def policy_action(state):
    state_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
    action_probs = policy_network(state_tensor).numpy().flatten()
    return np.argmax(action_probs)

test_episodes = 20
test_rewards = []

for episode in range(test_episodes):
    state = env.reset(seed=42)[0]
    total_reward = 0
    done = False

    while not done:
        action = policy_action(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    test_rewards.append(total_reward)

print(f"Average reward after training: {np.mean(test_rewards):.2f}")

# ===========================================
# 7. Visualize Training Performance
# ===========================================
plt.figure(figsize=(10, 6))
plt.plot(reward_history, label="Reward per Episode", color="blue")
plt.axhline(y=np.mean(test_rewards), color="red", linestyle="--", label="Average Test Reward")
plt.title("Policy Gradient Training Performance")
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
1. Policy Network: Outputs probabilities for actions.
2. Policy Gradient: Optimizes the policy directly using rewards.
3. Discounted Rewards: Ensures future rewards contribute less than immediate rewards.
4. Softmax Policy: Enables probabilistic action selection for exploration.

This script demonstrates how to implement the REINFORCE algorithm
to directly optimize policies for solving RL tasks.
"""
