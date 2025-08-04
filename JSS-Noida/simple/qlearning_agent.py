#!/usr/bin/env python3
"""
Q-Learning Agent
True agentic AI that learns from experience using reinforcement learning
"""

import random
import numpy as np

class QLearningAgent:
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.lr = learning_rate          # Learning rate (alpha)
        self.gamma = discount_factor     # Discount factor (gamma)
        self.epsilon = epsilon           # Exploration rate

        # Q-table: stores learned values for state-action pairs
        self.q_table = {}

        # Actions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        # Track learning progress
        self.episode_rewards = []
        self.episode_steps = []

    def get_state(self, agent_pos, goal_pos):
        """Convert positions to state representation."""
        return (agent_pos[0], agent_pos[1], goal_pos[0], goal_pos[1])

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore
        else:
            # Exploit: choose best action
            q_values = [self.get_q_value(state, a) for a in range(4)]
            if all(q == q_values[0] for q in q_values):
                return random.randint(0, 3)  # Random if all equal
            return q_values.index(max(q_values))

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula."""
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in range(4)])

        # Q-learning update rule
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def is_valid_move(self, pos, obstacles):
        """Check if position is valid."""
        return (0 <= pos[0] < self.grid_size and 
                0 <= pos[1] < self.grid_size and 
                pos not in obstacles)

    def get_reward(self, agent_pos, goal_pos, hit_obstacle=False):
        """Calculate reward for current situation."""
        if hit_obstacle:
            return -100  # Heavy penalty for obstacle
        elif agent_pos == goal_pos:
            return 100   # Big reward for goal
        else:
            # Small penalty for each step + distance penalty
            distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
            return -1 - distance * 0.5

    def train_episode(self, start_pos, goal_pos, obstacles, max_steps=100):
        """Train agent for one episode."""
        agent_pos = start_pos.copy()
        total_reward = 0
        steps = 0

        while agent_pos != goal_pos and steps < max_steps:
            state = self.get_state(agent_pos, goal_pos)
            action = self.choose_action(state, training=True)

            # Try to move
            new_pos = [agent_pos[0] + self.actions[action][0], 
                      agent_pos[1] + self.actions[action][1]]

            hit_obstacle = False
            if self.is_valid_move(new_pos, obstacles):
                agent_pos = new_pos
            else:
                hit_obstacle = True

            # Get reward and learn
            reward = self.get_reward(agent_pos, goal_pos, hit_obstacle)
            next_state = self.get_state(agent_pos, goal_pos)
            self.update_q_value(state, action, reward, next_state)

            total_reward += reward
            steps += 1

        # Record episode results
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)

        return agent_pos == goal_pos, steps, total_reward

    def test_run(self, start_pos, goal_pos, obstacles, max_steps=50):
        """Test the trained agent (no learning)."""
        agent_pos = start_pos.copy()
        path = [agent_pos.copy()]
        steps = 0

        while agent_pos != goal_pos and steps < max_steps:
            state = self.get_state(agent_pos, goal_pos)
            action = self.choose_action(state, training=False)

            new_pos = [agent_pos[0] + self.actions[action][0], 
                      agent_pos[1] + self.actions[action][1]]

            if self.is_valid_move(new_pos, obstacles):
                agent_pos = new_pos
                path.append(agent_pos.copy())
                steps += 1
            else:
                # Agent tried invalid move during testing
                break

        success = agent_pos == goal_pos
        return success, steps, path

    def get_policy(self, goal_pos):
        """Get the learned policy for visualization."""
        policy = {}
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = self.get_state([row, col], goal_pos)
                q_values = [self.get_q_value(state, a) for a in range(4)]
                best_action = q_values.index(max(q_values))
                policy[(row, col)] = self.action_names[best_action]
        return policy

# Test the Q-learning agent if run directly
if __name__ == "__main__":
    agent = QLearningAgent(5)
    obstacles = [[1, 1], [2, 2]]

    # Train
    print("Training...")
    for episode in range(100):
        agent.train_episode([0, 0], [4, 4], obstacles)

    # Test
    agent.epsilon = 0
    success, steps, path = agent.test_run([0, 0], [4, 4], obstacles)
    print(f"Q-Learning Agent: {'Success' if success else 'Failed'} in {steps} steps")
    print(f"Path: {path}")
    print(f"Q-table size: {len(agent.q_table)}")
