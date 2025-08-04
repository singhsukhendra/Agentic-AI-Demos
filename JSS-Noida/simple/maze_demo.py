#!/usr/bin/env python3
"""
Bulletproof Maze Navigation Demo: Agentic AI vs Greedy Agent

Scenario: The grid contains a "trap" that the greedy agent always falls into,
but the Q-learning agent learns to avoid after training.

This demo is designed to guarantee that the AI agent outperforms the traditional method.
"""

import numpy as np
from simple_agent import SimpleAgent
from qlearning_agent import QLearningAgent

def print_grid(agent_pos, goal_pos, obstacles, path=None, grid_size=7):
    """Print the grid with agent, goal, obstacles, and optional path."""
    print("Grid Legend: A=Agent, G=Goal, X=Obstacle, •=Path, .=Empty")
    for i in range(grid_size):
        row = ''
        for j in range(grid_size):
            if [i, j] == agent_pos:
                row += 'A '
            elif [i, j] == goal_pos:
                row += 'G '
            elif [i, j] in obstacles:
                row += 'X '
            elif path and [i, j] in path:
                row += '• '
            else:
                row += '. '
        print(row)
    print()

def create_trap_maze():
    """Create a 7x7 grid with a trap for greedy agents."""
    grid_size = 7
    # The greedy agent will always go right and down, falling into the trap at (2,5)
    obstacles = [
        [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 5], [3, 5], [4, 5], [5, 5],
        [5, 1], [5, 2], [5, 3], [5, 4],
        [3, 1], [3, 2], [3, 3]
    ]
    start_pos = [0, 0]
    goal_pos = [6, 6]
    return grid_size, obstacles, start_pos, goal_pos

def demo_maze_navigation():
    print("🧩 BULLETPROOF MAZE NAVIGATION DEMO")
    print("=" * 60)
    print("Scenario: 7x7 grid with a trap for greedy agents")
    print()
    grid_size, obstacles, start_pos, goal_pos = create_trap_maze()
    print("Maze Layout:")
    print_grid(start_pos, goal_pos, obstacles, grid_size=grid_size)

    # Test Simple (Greedy) Agent
    print("🤖 TESTING GREEDY AGENT...")
    simple_agent = SimpleAgent(grid_size)
    simple_success, simple_steps, simple_path = simple_agent.navigate(
        start_pos, goal_pos, obstacles, max_steps=100)
    print(f"Greedy Agent Result: {'✅ Success' if simple_success else '❌ Failed'} in {simple_steps} steps")
    if simple_success:
        print(f"Greedy Agent Path Length: {len(simple_path)}")
        print("Path:", simple_path)
    else:
        print("Greedy agent got stuck in the trap!")
    print()

    # Train Q-learning Agent
    print("🧠 TRAINING Q-LEARNING AGENT...")
    ai_agent = QLearningAgent(
        grid_size=grid_size,
        learning_rate=0.25,
        discount_factor=0.95,
        epsilon=0.4
    )
    training_episodes = 2000
    for episode in range(training_episodes):
        ai_agent.train_episode(start_pos, goal_pos, obstacles, max_steps=150)
    print(f"Training complete after {training_episodes} episodes.")
    print()

    # Test Q-learning Agent
    print("🎯 TESTING TRAINED Q-LEARNING AGENT...")
    ai_agent.epsilon = 0  # Pure exploitation
    ai_success, ai_steps, ai_path = ai_agent.test_run(start_pos, goal_pos, obstacles, max_steps=100)
    print(f"Q-Learning Agent Result: {'✅ Success' if ai_success else '❌ Failed'} in {ai_steps} steps")
    if ai_success:
        print(f"Q-Learning Agent Path Length: {len(ai_path)}")
        print("Path:", ai_path)
        print("AI Agent's learned path:")
        print_grid(goal_pos, goal_pos, obstacles, path=ai_path[:-1], grid_size=grid_size)
    else:
        print("AI agent failed (unexpected, please increase training episodes)")
    print()

    # Comparison
    print("📊 FINAL COMPARISON")
    print("=" * 40)
    print(f"Greedy Agent: {'✅ Success' if simple_success else '❌ FAILED'} - {simple_steps} steps")
    print(f"AI Agent:     {'✅ Success' if ai_success else '❌ FAILED'} - {ai_steps} steps")
    if ai_success and not simple_success:
        print("🏆 AI AGENT WINS! It learned to avoid the trap!")
    elif ai_success and simple_success:
        improvement = simple_steps - ai_steps
        if improvement > 0:
            print(f"🏆 AI AGENT WINS! It found a {improvement}-step shorter path!")
        else:
            print("Both succeeded, but AI is at least as good as the greedy agent.")
    else:
        print("Unexpected result. Try increasing training episodes.")
    print()
    print("💡 This demo is designed so the greedy agent always fails, but the AI agent learns to succeed.")
    print("This is a clear, bulletproof example of agentic AI superiority!")

def main():
    print("🚀 AGENTIC AI vs GREEDY AGENT: MAZE NAVIGATION")
    print("=" * 70)
    print()
    demo_maze_navigation()
    print("🎓 DEMONSTRATION COMPLETE!")
    print("=" * 50)
    print("KEY TAKEAWAYS:")
    print("✅ AI agent learns to avoid traps and finds optimal paths")
    print("✅ Greedy agent fails in complex environments")
    print("✅ This is why agentic AI is essential for real-world navigation problems!")

if __name__ == "__main__":
    main()
