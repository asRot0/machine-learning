import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Create the CartPole environment
env = gym.make("CartPole-v1")

# DQN Hyperparameters
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000  # Experience Replay Buffer
BATCH_SIZE = 32
EPSILON = 1.0  # Initial Exploration Rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Experience Replay Memory
memory = deque(maxlen=MEMORY_SIZE)

# Build the Deep Q-Network (DQN) model
def build_dqn():
    model = Sequential([
        Dense(24, activation='relu', input_shape=(STATE_SIZE,)),
        Dense(24, activation='relu'),
        Dense(ACTION_SIZE, activation='linear')  # Linear activation for Q-values
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Initialize the model
model = build_dqn()

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state):
    global EPSILON
    if np.random.rand() <= EPSILON:
        return random.randrange(ACTION_SIZE)  # Explore
    q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    return np.argmax(q_values[0])  # Exploit

def replay():
    if len(memory) < BATCH_SIZE:
        return
    minibatch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += GAMMA * np.max(model.predict(np.expand_dims(next_state, axis=0), verbose=0))
        target_f = model.predict(np.expand_dims(state, axis=0), verbose=0)
        target_f[0][action] = target
        model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
    global EPSILON
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

# Training Loop
EPISODES = 500
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    for time_step in range(200):  # Max time steps per episode
        action = act(state)
        next_state, reward, done, _, _ = env.step(action)
        remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    replay()
    print(f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {EPSILON:.4f}")

# Save the trained model
model.save("dqn_cartpole.h5")

# Test the trained model
def test_trained_model():
    trained_model = tf.keras.models.load_model("dqn_cartpole.h5")
    state, _ = env.reset()
    total_reward = 0
    for _ in range(200):
        action = np.argmax(trained_model.predict(np.expand_dims(state, axis=0), verbose=0))
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
        if done:
            break
    print(f"Test Run - Total Reward: {total_reward}")

test_trained_model()
env.close()
