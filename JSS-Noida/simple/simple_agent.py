#!/usr/bin/env python3
"""
Simple Rule-Based Agent
Traditional programming approach - follows hardcoded rules
"""

class SimpleAgent:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def is_valid_move(self, pos, obstacles):
        """Check if position is within bounds and not an obstacle."""
        return (0 <= pos[0] < self.grid_size and 
                0 <= pos[1] < self.grid_size and 
                pos not in obstacles)

    def get_possible_moves(self, agent_pos, obstacles):
        """Get all valid moves from current position."""
        moves = [
            [agent_pos[0] - 1, agent_pos[1]],  # Up
            [agent_pos[0] + 1, agent_pos[1]],  # Down
            [agent_pos[0], agent_pos[1] - 1],  # Left
            [agent_pos[0], agent_pos[1] + 1]   # Right
        ]

        return [move for move in moves if self.is_valid_move(move, obstacles)]

    def choose_best_move(self, possible_moves, goal_pos):
        """Choose move that minimizes Manhattan distance to goal."""
        if not possible_moves:
            return None

        best_move = possible_moves[0]
        best_distance = abs(best_move[0] - goal_pos[0]) + abs(best_move[1] - goal_pos[1])

        for move in possible_moves:
            distance = abs(move[0] - goal_pos[0]) + abs(move[1] - goal_pos[1])
            if distance < best_distance:
                best_distance = distance
                best_move = move

        return best_move

    def navigate(self, start_pos, goal_pos, obstacles, max_steps=50):
        """Navigate from start to goal using simple rules."""
        current_pos = start_pos.copy()
        path = [current_pos.copy()]
        steps = 0

        while current_pos != goal_pos and steps < max_steps:
            possible_moves = self.get_possible_moves(current_pos, obstacles)

            if not possible_moves:
                # Agent is stuck
                return False, steps, path

            next_pos = self.choose_best_move(possible_moves, goal_pos)

            if next_pos is None:
                return False, steps, path

            current_pos = next_pos
            path.append(current_pos.copy())
            steps += 1

        success = current_pos == goal_pos
        return success, steps, path

# Test the simple agent if run directly
if __name__ == "__main__":
    agent = SimpleAgent(5)
    obstacles = [[1, 1], [2, 2]]
    success, steps, path = agent.navigate([0, 0], [4, 4], obstacles)
    print(f"Simple Agent: {'Success' if success else 'Failed'} in {steps} steps")
    print(f"Path: {path}")
