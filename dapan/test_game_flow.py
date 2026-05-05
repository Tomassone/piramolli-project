import random

from game import Game


def main():
    game = Game()
    state = game.get_initial_state()

    print("Initial state loaded.")
    print("Terminal?", game.is_terminal(state))

    max_steps = 10

    for step in range(max_steps):
        valid_moves = game.get_valid_moves(state)

        print(f"\nStep {step}")
        print(f"Number of valid moves: {len(valid_moves)}")

        if len(valid_moves) == 0:
            print("No valid moves available.")
            break

        action = random.choice(valid_moves)
        print(f"Chosen action: {action}")

        next_state = game.get_next_state(state, action)
        print("Applied action successfully.")

        if game.is_terminal(next_state):
            print("Reached terminal state.")
            winner = game.get_winner(next_state)
            print(f"Winner: {winner}")
            break

        state = next_state

    print("\nTest completed.")


if __name__ == "__main__":
    main()