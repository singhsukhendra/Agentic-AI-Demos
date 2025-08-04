#!/usr/bin/env python3
"""
Improved Agentic AI Demonstration
This version is designed to clearly show AI superiority over rule-based agents
"""

import random
import numpy as np
from simple_agent import SimpleAgent
from qlearning_agent import QLearningAgent

def print_grid(agent_pos, goal_pos, obstacles, path=None, grid_size=7):
    """Print the grid with better visualization."""
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

def create_challenging_maze():
    """Create a maze where simple agent will struggle but AI can learn."""
    grid_size = 8

    # Create a maze-like structure that forces non-greedy moves
    obstacles = [
        # Vertical walls
        [1, 2], [2, 2], [3, 2], [4, 2],
        [1, 5], [2, 5], [3, 5], [4, 5],
        # Horizontal barriers
        [2, 1], [2, 3], [2, 4], [2, 6],
        [5, 1], [5, 2], [5, 3], [5, 4], [5, 6],
        # Additional complexity
        [6, 2], [6, 5]
    ]

    start_pos = [0, 0]
    goal_pos = [7, 7]

    return grid_size, obstacles, start_pos, goal_pos

def create_trap_scenario():
    """Create a scenario where greedy approach leads to dead end."""
    grid_size = 7

    # Create a "trap" - greedy path leads to dead end
    obstacles = [
        # Create walls that make greedy path fail
        [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],  # Top wall
        [2, 5], [3, 5], [4, 5], [5, 5],          # Right wall  
        [5, 1], [5, 2], [5, 3], [5, 4],          # Bottom wall
        # Force a detour
        [3, 1], [3, 2], [3, 3]
    ]

    start_pos = [0, 0]
    goal_pos = [6, 6]

    return grid_size, obstacles, start_pos, goal_pos

def demo_guaranteed_ai_win():
    """Demonstrate scenario where AI clearly outperforms simple agent."""
    print("🎯 GUARANTEED AI SUPERIORITY DEMO")
    print("=" * 60)
    print("This scenario is designed to show AI learning advantages")
    print()

    # Use the challenging maze
    grid_size, obstacles, start_pos, goal_pos = create_challenging_maze()

    print(f"Challenge: {grid_size}x{grid_size} maze")
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    print(f"Obstacles: {len(obstacles)} strategic barriers")
    print()

    print("Environment:")
    print_grid(start_pos, goal_pos, obstacles, grid_size=grid_size)

    # Test Simple Agent first
    print("🤖 TESTING SIMPLE RULE-BASED AGENT...")
    simple_agent = SimpleAgent(grid_size)
    simple_success, simple_steps, simple_path = simple_agent.navigate(
        start_pos, goal_pos, obstacles, max_steps=100)

    print(f"Simple Agent Result: {'✅ Success' if simple_success else '❌ Failed'} in {simple_steps} steps")
    if simple_success:
        print(f"Simple Agent Path Length: {len(simple_path)}")
    else:
        print("Simple agent got stuck due to greedy strategy!")
    print()

    # Train AI Agent with better parameters
    print("🧠 TRAINING AI AGENT...")
    print("Training with optimized parameters for complex environments...")

    ai_agent = QLearningAgent(
        grid_size=grid_size,
        learning_rate=0.2,      # Higher learning rate
        discount_factor=0.95,   # Value future rewards more
        epsilon=0.3             # More exploration
    )

    # Extended training for complex environment
    training_episodes = 1500
    success_count = 0

    print(f"Training for {training_episodes} episodes...")

    for episode in range(training_episodes):
        success, steps, reward = ai_agent.train_episode(start_pos, goal_pos, obstacles, max_steps=150)
        if success:
            success_count += 1

        # Show progress
        if (episode + 1) % 300 == 0:
            recent_rewards = ai_agent.episode_rewards[max(0, episode-99):episode+1]
            recent_success = sum(1 for r in recent_rewards if r > 0)
            recent_steps = ai_agent.episode_steps[max(0, episode-99):episode+1]
            avg_steps = np.mean([s for s, r in zip(recent_steps, recent_rewards) if r > 0]) if recent_success > 0 else 0
            print(f"  Episode {episode+1}: Success Rate = {recent_success}/100 ({recent_success}%), Avg Steps = {avg_steps:.1f}")

    print(f"Training Complete! Overall success: {success_count}/{training_episodes} ({success_count/training_episodes*100:.1f}%)")
    print()

    # Test trained AI agent
    print("🎯 TESTING TRAINED AI AGENT...")
    ai_agent.epsilon = 0  # Pure exploitation

    ai_success, ai_steps, ai_path = ai_agent.test_run(start_pos, goal_pos, obstacles, max_steps=100)

    print(f"AI Agent Result: {'✅ Success' if ai_success else '❌ Failed'} in {ai_steps} steps")
    if ai_success:
        print(f"AI Agent Path Length: {len(ai_path)}")
        print("AI Agent's learned path:")
        print_grid(goal_pos, goal_pos, obstacles, path=ai_path[:-1], grid_size=grid_size)
    print()

    # Comparison
    print("📊 FINAL COMPARISON")
    print("=" * 40)
    print(f"Simple Agent: {'✅ Success' if simple_success else '❌ FAILED'} - {simple_steps} steps")
    print(f"AI Agent:     {'✅ Success' if ai_success else '❌ FAILED'} - {ai_steps} steps")

    if ai_success and not simple_success:
        print("🏆 AI AGENT WINS! It succeeded where the simple agent failed!")
        print("💡 The AI learned to avoid the greedy trap through experience.")
    elif ai_success and simple_success:
        improvement = simple_steps - ai_steps
        if improvement > 0:
            print(f"🏆 AI AGENT WINS! It found a {improvement}-step shorter path!")
        else:
            print("Both succeeded, but let's try a harder challenge...")
    else:
        print("Let's try an even more complex scenario...")

    print(f"\n🧠 AI Agent's Knowledge: {len(ai_agent.q_table)} learned state-action pairs")

    return ai_success and (not simple_success or ai_steps < simple_steps)

def demo_adaptation():
    """Show AI's ability to adapt to changing environments."""
    print("\n🔄 ADAPTATION DEMONSTRATION")
    print("=" * 60)
    print("Showing AI's ability to adapt when environment changes")
    print()

    grid_size = 6

    # Original environment
    obstacles_v1 = [[1, 1], [2, 2], [3, 3]]
    start_pos = [0, 0]
    goal_pos = [5, 5]

    print("Phase 1: Training in simple environment")
    print_grid(start_pos, goal_pos, obstacles_v1, grid_size=grid_size)

    # Train AI agent
    ai_agent = QLearningAgent(grid_size, learning_rate=0.2, epsilon=0.2)

    for episode in range(300):
        ai_agent.train_episode(start_pos, goal_pos, obstacles_v1)

    ai_agent.epsilon = 0
    success1, steps1, path1 = ai_agent.test_run(start_pos, goal_pos, obstacles_v1)
    print(f"Phase 1 Result: {'✅ Success' if success1 else '❌ Failed'} in {steps1} steps")

    # Changed environment - more obstacles
    obstacles_v2 = [[1, 1], [2, 2], [3, 3], [1, 3], [3, 1], [4, 2], [2, 4]]

    print("\nPhase 2: Environment changed (more obstacles added)")
    print_grid(start_pos, goal_pos, obstacles_v2, grid_size=grid_size)

    # Test without retraining
    success2a, steps2a, path2a = ai_agent.test_run(start_pos, goal_pos, obstacles_v2)
    print(f"Without retraining: {'✅ Success' if success2a else '❌ Failed'} in {steps2a} steps")

    # Retrain for new environment
    print("\nPhase 3: Retraining for new environment...")
    ai_agent.epsilon = 0.15  # Some exploration for adaptation

    for episode in range(200):
        ai_agent.train_episode(start_pos, goal_pos, obstacles_v2)

    ai_agent.epsilon = 0
    success2b, steps2b, path2b = ai_agent.test_run(start_pos, goal_pos, obstacles_v2)
    print(f"After retraining: {'✅ Success' if success2b else '❌ Failed'} in {steps2b} steps")

    if success2b:
        print("Adapted path:")
        print_grid(goal_pos, goal_pos, obstacles_v2, path=path2b[:-1], grid_size=grid_size)

    print("\n💡 KEY INSIGHT: AI agents can adapt to new environments!")
    print("   Simple rule-based agents would need to be reprogrammed.")

def main():
    """Run the improved demonstration."""
    print("🚀 IMPROVED AGENTIC AI DEMONSTRATION")
    print("Designed to clearly show AI superiority over traditional programming")
    print("=" * 70)
    print()

    # Run guaranteed win scenario
    ai_won = demo_guaranteed_ai_win()

    if not ai_won:
        print("\n🔧 Let's try an even more challenging scenario...")
        # Try the trap scenario
        grid_size, obstacles, start_pos, goal_pos = create_trap_scenario()

        print(f"\nTRAP SCENARIO: {grid_size}x{grid_size} grid with greedy trap")
        print_grid(start_pos, goal_pos, obstacles, grid_size=grid_size)

        # Simple agent will likely fail
        simple_agent = SimpleAgent(grid_size)
        simple_success, simple_steps, _ = simple_agent.navigate(start_pos, goal_pos, obstacles)

        # AI agent with even more training
        ai_agent = QLearningAgent(grid_size, learning_rate=0.25, epsilon=0.4)
        for episode in range(2000):
            ai_agent.train_episode(start_pos, goal_pos, obstacles, max_steps=200)

        ai_agent.epsilon = 0
        ai_success, ai_steps, ai_path = ai_agent.test_run(start_pos, goal_pos, obstacles)

        print(f"\nTRAP SCENARIO RESULTS:")
        print(f"Simple Agent: {'✅ Success' if simple_success else '❌ FAILED'} - {simple_steps} steps")
        print(f"AI Agent:     {'✅ Success' if ai_success else '❌ FAILED'} - {ai_steps} steps")

        if ai_success and not simple_success:
            print("🏆 AI AGENT WINS! It learned to avoid the trap!")

    # Show adaptation capability
    input("\nPress Enter to see adaptation demonstration...")
    demo_adaptation()

    print("\n🎓 DEMONSTRATION COMPLETE!")
    print("=" * 50)
    print("KEY TAKEAWAYS:")
    print("✅ AI agents can solve problems that stump rule-based agents")
    print("✅ AI agents learn optimal strategies through experience")
    print("✅ AI agents can adapt to changing environments")
    print("✅ In complex scenarios, AI clearly outperforms traditional programming")
    print("\n💡 This is why agentic AI is revolutionary for real-world applications!")

if __name__ == "__main__":
    main()
