from dapan.game import Game
import re

with open('dapan/game.py', 'r') as f:
    content = f.read()

# Update get_initial_state to track move clock, etc.
init_state_new = """    def get_initial_state(self):
        return {
                'white_positions': [
                    (2, 4), (3, 4), (4, 2), (4, 3), 
                    (4, 5), (4, 6), (5, 4), (6, 4)
                ],
                'black_positions': [
                    (0, 3), (0, 4), (0, 5), (1, 4),
                    (3, 0), (4, 0), (5, 0), (4, 1),
                    (8, 3), (8, 4), (8, 5), (7, 4),
                    (3, 8), (4, 8), (5, 8), (4, 7)
                ],
                'king_position': (4, 4),
                'turn_to_move': 1,
                'move_count': 0,
                'half_move_clock': 0
            }"""

content = re.sub(r'    def get_initial_state\(self\):.*?turn_to_move\': 1\n            }', init_state_new, content, flags=re.DOTALL)

with open('dapan/game_patched.py', 'w') as f:
    f.write(content)
