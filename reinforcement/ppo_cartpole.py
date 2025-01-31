"""
Proximal Policy Optimization (PPO) Implementation
==================================================
This script implements PPO to solve the CartPole-v1 environment using TensorFlow.

PPO is an advanced policy gradient method that addresses stability and efficiency issues in RL training. It uses a
"surrogate" loss function to constrain policy updates, ensuring balanced exploration and exploitation.
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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
# 2. Define Actor and Critic Networks
# ===========================================
def build_actor(n_states, n_actions):
    model = tf.keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(n_states,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_actions, activation="softmax")  # Output probabilities
    ])
    return model

def build_critic(n_states):
    model = tf.keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(n_states,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)  # Output value estimate
    ])
    return model

# Initialize networks
actor = build_actor(n_states, n_actions)
critic = build_critic(n_states)

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

# ===========================================
# 3. Define PPO Loss Functions
# ===========================================
def compute_actor_loss(states, actions, advantages, old_probs, epsilon=0.2):
    probabilities = actor(states, training=True)
    action_probs = tf.gather(probabilities, actions, batch_dims=1, axis=1)

    # Ratio for clipped surrogate objective
    ratios = action_probs / (old_probs + 1e-8)
    clipped_ratios = tf.clip_by_value(ratios, 1 - epsilon, 1 + epsilon)
    surrogate_loss = tf.minimum(ratios * advantages, clipped_ratios * advantages)

    return -tf.reduce_mean(surrogate_loss)  # Negative because we maximize reward

def compute_critic_loss(values, returns):
    return tf.reduce_mean(tf.square(returns - values))  # Mean squared error

# ===========================================
# 4. Compute Advantage and Returns
# ===========================================
def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    deltas = rewards + gamma * np.append(values[1:], 0) - values[:-1]
    advantages = []
    advantage = 0
    for delta in reversed(deltas):
        advantage = delta + gamma * lambda_ * advantage
        advantages.insert(0, advantage)
    return np.array(advantages)

def compute_discounted_returns(rewards, gamma=0.99):
    discounted_returns = np.zeros_like(rewards, dtype=np.float32)
    cumulative_return = 0.0
    for t in reversed(range(len(rewards))):
        cumulative_return = rewards[t] + gamma * cumulative_return
        discounted_returns[t] = cumulative_return
    return discounted_returns

# ===========================================
# 5. Train PPO Agent
# ===========================================
episodes = 1000
batch_size = 32
gamma = 0.99
epsilon = 0.2
lambda_ = 0.95
reward_history = []

for episode in range(episodes):
    states, actions, rewards, old_probs, values = [], [], [], [], []
    state = env.reset(seed=42)[0]
    done = False
    total_reward = 0

    # Collect trajectory
    while not done:
        state_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        probabilities = actor(state_tensor).numpy().flatten()
        action = np.random.choice(n_actions, p=probabilities)

        next_state, reward, done, _, _ = env.step(action)

        # Store trajectory
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        old_probs.append(probabilities[action])
        values.append(critic(state_tensor).numpy().flatten()[0])

        state = next_state
        total_reward += reward

    # Post-episode processing
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

    returns = compute_discounted_returns(rewards, gamma)
    advantages = compute_advantages(rewards, values, gamma, lambda_)

    # Update networks
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        critic_values = critic(states, training=True)
        actor_loss = compute_actor_loss(states, actions, advantages, old_probs, epsilon)
        critic_loss = compute_critic_loss(critic_values, returns)

    actor_gradients = actor_tape.gradient(actor_loss, actor.trainable_variables)
    critic_gradients = critic_tape.gradient(critic_loss, critic.trainable_variables)

    actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

    reward_history.append(total_reward)
    print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward:.2f} | "
          f"Actor Loss: {actor_loss.numpy():.4f} | Critic Loss: {critic_loss.numpy():.4f}")

# ===========================================
# 6. Evaluate Trained PPO Agent
# ===========================================
test_episodes = 20
test_rewards = []

for episode in range(test_episodes):
    state = env.reset(seed=42)[0]
    total_reward = 0
    done = False

    while not done:
        state_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        action_probs = actor(state_tensor).numpy().flatten()
        action = np.argmax(action_probs)

        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    test_rewards.append(total_reward)

print(f"Average test reward after training: {np.mean(test_rewards):.2f}")

# ===========================================
# 7. Visualize Training Performance
# ===========================================
plt.figure(figsize=(10, 6))
plt.plot(reward_history, label="Reward per Episode", color="blue")
plt.axhline(y=np.mean(test_rewards), color="red", linestyle="--", label="Average Test Reward")
plt.title("PPO Training Performance")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.show()

# ===========================================
# Summary
# ===========================================
"""
Key Concepts in PPO:
1. Actor-Critic Framework: Actor chooses actions; Critic evaluates them.
2. Clipped Surrogate Objective: Stabilizes training by constraining policy updates.
3. Generalized Advantage Estimation (GAE): Smooths advantage calculations.
4. Parallel Training: Collects trajectories and updates in batches for efficiency.

PPO strikes a balance between simplicity, stability, and performance in RL tasks.
"""
