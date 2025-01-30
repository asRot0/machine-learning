"""
Introduction to Reinforcement Learning
=======================================
This script demonstrates the fundamental concepts of reinforcement learning (RL)
using OpenAI Gym's CartPole environment.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

# ===========================================
# 1. Define the Environment
# ===========================================
# Create the CartPole environment
env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f"Number of states: {n_states}")
print(f"Number of actions: {n_actions}")

# ===========================================
# 2. Define a Random Policy
# ===========================================
# A random policy selects actions randomly
def random_policy(state):
    return env.action_space.sample()

# ===========================================
# 3. Simulate the Environment with the Policy
# ===========================================
def run_episode(env, policy, render=False):
    state = env.reset(seed=42)
    total_reward = 0
    done = False

    while not done:
        if render:
            env.render()

        # Select an action using the policy
        action = policy(state)

        # Apply the action to the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

# Run a single episode with the random policy
reward = run_episode(env, random_policy)
print(f"Total reward from a random policy: {reward}")

# ===========================================
# 4. Evaluate the Random Policy
# ===========================================
# Evaluate the policy over multiple episodes
n_episodes = 100
rewards = [run_episode(env, random_policy) for _ in range(n_episodes)]

# Plot the rewards
plt.figure(figsize=(10, 6))
plt.hist(rewards, bins=20, color="skyblue", edgecolor="black")
plt.title("Rewards Distribution with Random Policy")
plt.xlabel("Total Reward")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ===========================================
# 5. Implement a Basic Q-Learning Algorithm
# ===========================================
# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
n_episodes = 1000

# Initialize the Q-table
q_table = np.zeros((20, 20, n_actions))

# Function to discretize the continuous state space
def discretize_state(state):
    bins = [np.linspace(-4.8, 4.8, 20),  # Cart position
            np.linspace(-5, 5, 20),      # Cart velocity
            np.linspace(-0.418, 0.418, 20),  # Pole angle
            np.linspace(-5, 5, 20)]      # Pole velocity
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))

# Q-learning implementation
for episode in range(n_episodes):
    state = env.reset(seed=42)
    state = discretize_state(state[0])
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Perform the action
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        total_reward += reward

        # Update Q-value
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + discount_factor * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += learning_rate * td_error

        state = next_state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Log progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{n_episodes}: Total Reward = {total_reward}")

# ===========================================
# 6. Test the Trained Q-Learning Agent
# ===========================================
def q_learning_policy(state):
    state = discretize_state(state)
    return np.argmax(q_table[state])

test_rewards = [run_episode(env, q_learning_policy) for _ in range(100)]

print(f"Average reward after training: {np.mean(test_rewards)}")

# ===========================================
# 7. Visualize the Q-Learning Performance
# ===========================================
plt.figure(figsize=(10, 6))
plt.plot(test_rewards, label="Rewards per Episode", color="green")
plt.axhline(y=np.mean(test_rewards), color="red", linestyle="--", label="Average Reward")
plt.title("Performance of Q-Learning Agent")
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
1. Basics of Reinforcement Learning: states, actions, rewards, policies.
2. Random Policy: A simple baseline.
3. Q-Learning Algorithm: A foundational RL method.
4. Discretization of Continuous State Space: Making Q-learning feasible for CartPole.
5. Evaluation: Comparing random policy vs. trained Q-learning agent.
"""
