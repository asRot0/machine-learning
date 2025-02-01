"""
Deep Deterministic Policy Gradient (DDPG) for Robotic Arm Control
==================================================================
This script demonstrates how to use Deep Deterministic Policy Gradient (DDPG) to control a robotic arm in a simulation
environment powered by PyBullet. DDPG is well-suited for continuous action spaces like robotic control tasks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import time

# ===========================================
# 1. Define the Environment
# ===========================================
class RoboticArmEnv:
    def __init__(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load robotic arm and plane
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        # Define action and observation spaces
        self.n_actions = p.getNumJoints(self.robot)
        self.n_states = self.n_actions * 2

    def reset(self):
        for joint in range(self.n_actions):
            p.resetJointState(self.robot, joint, targetValue=0, targetVelocity=0)
        return self.get_state()

    def get_state(self):
        state = []
        for joint in range(self.n_actions):
            joint_state = p.getJointState(self.robot, joint)
            state.extend([joint_state[0], joint_state[1]])  # Position and velocity
        return np.array(state, dtype=np.float32)

    def step(self, action):
        for joint in range(self.n_actions):
            p.setJointMotorControl2(
                self.robot,
                joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action[joint],
                force=500
            )
        p.stepSimulation()
        time.sleep(0.01)

        # Compute reward and done
        state = self.get_state()
        reward = -np.linalg.norm(state[:self.n_actions])  # Reward based on arm position
        done = False  # Add termination logic if needed

        return state, reward, done, {}

    def close(self):
        p.disconnect()

# Initialize environment
env = RoboticArmEnv()

# ===========================================
# 2. Define Actor and Critic Networks
# ===========================================
def build_actor(n_states, n_actions, action_bounds):
    model = tf.keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(n_states,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_actions, activation="tanh"),
        layers.Lambda(lambda x: x * action_bounds)  # Scale output to action bounds
    ])
    return model

def build_critic(n_states, n_actions):
    state_input = layers.Input(shape=(n_states,))
    action_input = layers.Input(shape=(n_actions,))
    concatenated = layers.Concatenate()([state_input, action_input])

    x = layers.Dense(64, activation="relu")(concatenated)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1)(x)

    return tf.keras.Model([state_input, action_input], output)

# Initialize networks
action_bounds = 1.0
actor = build_actor(env.n_states, env.n_actions, action_bounds)
critic = build_critic(env.n_states, env.n_actions)

target_actor = build_actor(env.n_states, env.n_actions, action_bounds)
target_critic = build_critic(env.n_states, env.n_actions)

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

# ===========================================
# 3. Define Replay Buffer
# ===========================================
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0

    def store(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

buffer = ReplayBuffer()

# ===========================================
# 4. Define DDPG Training Functions
# ===========================================
def update_target_network(target, source, tau=0.005):
    for target_var, source_var in zip(target.variables, source.variables):
        target_var.assign(tau * source_var + (1 - tau) * target_var)

def compute_critic_loss(states, actions, rewards, next_states, dones, gamma=0.99):
    target_actions = target_actor(next_states, training=True)
    target_q_values = target_critic([next_states, target_actions], training=True)
    y = rewards + gamma * (1 - dones) * target_q_values
    critic_q_values = critic([states, actions], training=True)
    return tf.reduce_mean(tf.square(y - critic_q_values))

def compute_actor_loss(states):
    actions = actor(states, training=True)
    q_values = critic([states, actions], training=True)
    return -tf.reduce_mean(q_values)  # Negative because we maximize Q-value

# ===========================================
# 5. Train DDPG Agent
# ===========================================
episodes = 500
batch_size = 64
gamma = 0.99
tau = 0.005
reward_history = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Select action with exploration noise
        action = actor(tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)).numpy().flatten()
        action += np.random.normal(scale=0.1, size=action.shape)  # Exploration noise

        # Execute action
        next_state, reward, done, _ = env.step(action)
        buffer.store((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Train from buffer if enough samples are collected
        if len(buffer.buffer) > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # Update critic
            with tf.GradientTape() as tape:
                critic_loss = compute_critic_loss(states, actions, rewards, next_states, dones, gamma)
            critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

            # Update actor
            with tf.GradientTape() as tape:
                actor_loss = compute_actor_loss(states)
            actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))

            # Update target networks
            update_target_network(target_actor, actor, tau)
            update_target_network(target_critic, critic, tau)

    reward_history.append(total_reward)
    print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward:.2f}")

env.close()

# ===========================================
# 6. Visualize Training Performance
# ===========================================
plt.figure(figsize=(10, 6))
plt.plot(reward_history, label="Reward per Episode", color="blue")
plt.title("DDPG Training Performance")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.show()


'''
PyBullet Simulation: Offers a physics engine and robotic environments.
DDPG Algorithm: Handles continuous action spaces using Actor-Critic.
Replay Buffer: Improves sample efficiency.
Target Networks: Stabilize training by slowly updating weights.
'''