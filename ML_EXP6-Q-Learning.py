"""
Q-Learning on a Maze Environment

- Maze is a grid with walls (1), free cells (0), a start (S) and goal (G).
- Agent receives:
    - reward = -0.1 for each step (encourages shorter paths)
    - reward = -1.0 if it tries to move into a wall (penalty)
    - reward = +10 on reaching goal (episode ends)
- Q-learning with epsilon-greedy exploration.
- At the end it prints learned policy and a simple heatmap of state values.

Run: python q_learning_maze.py
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------
# Maze definition
# -------------------------
# 0 = free cell
# 1 = wall
# We'll mark start and goal separately.
MAZE = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0],
])

START = (0, 0)
GOAL = (4, 5)

# Actions: up, right, down, left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_NAMES = ["↑", "→", "↓", "←"]

# -------------------------
# Environment functions
# -------------------------
def in_bounds(pos, maze):
    r, c = pos
    return 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1]

def is_free(pos, maze):
    r, c = pos
    return in_bounds(pos, maze) and maze[r, c] == 0

def step(state, action, maze):
    """Take action from state. Return new_state, reward, done."""
    r, c = state
    dr, dc = action
    new_pos = (r + dr, c + dc)
    # Hitting wall or out of bounds
    if not in_bounds(new_pos, maze) or maze[new_pos] == 1:
        # stay in same state, penalty
        return state, -1.0, False
    # moved to a free cell
    if new_pos == GOAL:
        return new_pos, 10.0, True
    # normal step penalty
    return new_pos, -0.1, False

# Map state (r,c) to index and back
state_to_idx = {}
idx_to_state = []
for r in range(MAZE.shape[0]):
    for c in range(MAZE.shape[1]):
        if MAZE[r, c] == 0:  # only free cells are valid states
            idx = len(idx_to_state)
            state_to_idx[(r, c)] = idx
            idx_to_state.append((r, c))
NUM_STATES = len(idx_to_state)
NUM_ACTIONS = len(ACTIONS)

# -------------------------
# Q-learning parameters
# -------------------------
np.random.seed(42)
random.seed(42)
alpha = 0.7         # learning rate
gamma = 0.95        # discount factor
epsilon = 0.2       # exploration rate
min_epsilon = 0.01
decay = 0.995       # epsilon decay per episode
episodes = 3000
max_steps_per_episode = 200

# Initialize Q-table
Q = np.zeros((NUM_STATES, NUM_ACTIONS))

# -------------------------
# Helper to choose action
# -------------------------
def choose_action(state_idx, eps):
    if np.random.rand() < eps:
        return np.random.randint(NUM_ACTIONS)
    return np.argmax(Q[state_idx])

# -------------------------
# Training loop
# -------------------------
episode_lengths = []
episode_rewards = []

for ep in range(episodes):
    state = START
    state_idx = state_to_idx[state]
    total_reward = 0.0
    for step_i in range(max_steps_per_episode):
        action_idx = choose_action(state_idx, epsilon)
        action = ACTIONS[action_idx]
        new_state, reward, done = step(state, action, MAZE)
        new_state_idx = state_to_idx[new_state]
        # Q-learning update
        best_next = np.max(Q[new_state_idx])
        Q[state_idx, action_idx] += alpha * (reward + gamma * best_next - Q[state_idx, action_idx])
        state = new_state
        state_idx = new_state_idx
        total_reward += reward
        if done:
            break
    # decay epsilon
    epsilon = max(min_epsilon, epsilon * decay)
    episode_lengths.append(step_i + 1)
    episode_rewards.append(total_reward)
    # Optional: print progress occasionally
    if (ep + 1) % 500 == 0:
        print(f"Episode {ep+1}/{episodes} — steps: {step_i+1} total_reward: {total_reward:.2f} eps: {epsilon:.3f}")

# -------------------------
# Derive policy and state values
# -------------------------
policy = np.full(MAZE.shape, " ")
value_grid = np.full(MAZE.shape, np.nan)

for (r, c), s_idx in state_to_idx.items():
    best_a = np.argmax(Q[s_idx])
    policy[r, c] = ACTION_NAMES[best_a]
    value_grid[r, c] = np.max(Q[s_idx])

policy[START] = "S"
policy[GOAL] = "G"

# Mark walls
for r in range(MAZE.shape[0]):
    for c in range(MAZE.shape[1]):
        if MAZE[r, c] == 1:
            policy[r, c] = "█"

# -------------------------
# Display results
# -------------------------
print("\nLearned policy grid (arrows show best action):")
for row in policy:
    print(" ".join(row))

print("\nSample Q-values for START state:")
start_idx = state_to_idx[START]
for i, a in enumerate(ACTION_NAMES):
    print(f"  {a}: {Q[start_idx, i]:.3f}")

# Plotting training reward curve
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'))
plt.title("Smoothed Episode Reward (window=50)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

# Plot value heatmap
plt.subplot(1,2,2)
plt.title("State Value Estimates (max Q)")
plt.imshow(value_grid, interpolation='nearest')
plt.colorbar(label='Value')
plt.scatter([START[1]], [START[0]], marker='o')  # start as circle
plt.text(START[1], START[0], 'S', color='white', ha='center', va='center', fontsize=12, weight='bold')
plt.scatter([GOAL[1]], [GOAL[0]], marker='*')    # goal as star
plt.text(GOAL[1], GOAL[0], 'G', color='white', ha='center', va='center', fontsize=12, weight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Simulate one greedy run from START following learned policy
print("\nOne greedy run from START following learned policy:")
state = START
path = [state]
for _ in range(100):
    s_idx = state_to_idx[state]
    action_idx = np.argmax(Q[s_idx])
    dr, dc = ACTIONS[action_idx]
    new_state, reward, done = step(state, (dr,dc), MAZE)
    path.append(new_state)
    state = new_state
    if done:
        break

print("Path:", path)
