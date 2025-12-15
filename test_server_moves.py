"""
Test script to verify if the server is processing moves correctly.
Makes a series of moves and checks if pieces actually move on the server.
"""
import requests
import json
from game import KingCapture, Player, Piece

def test_server_moves():
    """Test if server processes moves correctly."""
    game = KingCapture()
    game.USE_SERVER = True
    
    print("=" * 60)
    print("INITIAL STATE")
    print("=" * 60)
    print(f"White king: {game.white_king_pos}")
    print(f"White kinglike: {game.white_kinglike_pos}")
    print(f"Black king: {game.black_king_pos}")
    print(f"Black kinglike: {game.black_kinglike_pos}")
    print(f"Current player: {game.current_player}")
    print()
    
    # Make move 1: White moves kinglike forward
    print("=" * 60)
    print("MOVE 1: White moves kinglike (piece_idx=1) from (4,1) to (3,1)")
    print("=" * 60)
    print(f"Before move - White kinglike at: {game.white_kinglike_pos}")
    
    success = game.make_move(piece_idx=1, row=3, col=1)
    print(f"Move successful: {success}")
    print(f"After move - White kinglike at: {game.white_kinglike_pos}")
    print(f"After move - White king at: {game.white_king_pos}")
    print(f"After move - Black king at: {game.black_king_pos}")
    print(f"After move - Black kinglike at: {game.black_kinglike_pos}")
    print(f"Current player: {game.current_player}")
    print()
    
    # Make move 2: Black moves king forward
    print("=" * 60)
    print("MOVE 2: Black moves king (piece_idx=0) from (0,2) to (1,2)")
    print("=" * 60)
    print(f"Before move - Black king at: {game.black_king_pos}")
    
    success = game.make_move(piece_idx=0, row=1, col=2)
    print(f"Move successful: {success}")
    print(f"After move - White kinglike at: {game.white_kinglike_pos}")
    print(f"After move - White king at: {game.white_king_pos}")
    print(f"After move - Black king at: {game.black_king_pos}")
    print(f"After move - Black kinglike at: {game.black_kinglike_pos}")
    print(f"Current player: {game.current_player}")
    print()
    
    # Make move 3: White moves king forward
    print("=" * 60)
    print("MOVE 3: White moves king (piece_idx=0) from (4,2) to (3,2)")
    print("=" * 60)
    print(f"Before move - White king at: {game.white_king_pos}")
    
    success = game.make_move(piece_idx=0, row=3, col=2)
    print(f"Move successful: {success}")
    print(f"After move - White kinglike at: {game.white_kinglike_pos}")
    print(f"After move - White king at: {game.white_king_pos}")
    print(f"After move - Black king at: {game.black_king_pos}")
    print(f"After move - Black kinglike at: {game.black_kinglike_pos}")
    print(f"Current player: {game.current_player}")
    print()
    
    # Make move 4: Black moves kinglike forward
    print("=" * 60)
    print("MOVE 4: Black moves kinglike (piece_idx=1) from (0,1) to (1,1)")
    print("=" * 60)
    print(f"Before move - Black kinglike at: {game.black_kinglike_pos}")
    
    success = game.make_move(piece_idx=1, row=1, col=1)
    print(f"Move successful: {success}")
    print(f"After move - White kinglike at: {game.white_kinglike_pos}")
    print(f"After move - White king at: {game.white_king_pos}")
    print(f"After move - Black king at: {game.black_king_pos}")
    print(f"After move - Black kinglike at: {game.black_kinglike_pos}")
    print(f"Current player: {game.current_player}")
    print()
    
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"White king: {game.white_king_pos}")
    print(f"White kinglike: {game.white_kinglike_pos}")
    print(f"Black king: {game.black_king_pos}")
    print(f"Black kinglike: {game.black_kinglike_pos}")
    print(f"Move history: {game.move_history}")
    print()
    
    # Check if pieces actually moved
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    expected_white_king = (3, 2)  # Should have moved from (4,2) to (3,2)
    expected_white_kinglike = (3, 1)  # Should have moved from (4,1) to (3,1)
    expected_black_king = (1, 2)  # Should have moved from (0,2) to (1,2)
    expected_black_kinglike = (1, 1)  # Should have moved from (0,1) to (1,1)
    
    print(f"Expected white king: {expected_white_king}, Actual: {game.white_king_pos}, Match: {game.white_king_pos == expected_white_king}")
    print(f"Expected white kinglike: {expected_white_kinglike}, Actual: {game.white_kinglike_pos}, Match: {game.white_kinglike_pos == expected_white_kinglike}")
    print(f"Expected black king: {expected_black_king}, Actual: {game.black_king_pos}, Match: {game.black_king_pos == expected_black_king}")
    print(f"Expected black kinglike: {expected_black_kinglike}, Actual: {game.black_kinglike_pos}, Match: {game.black_kinglike_pos == expected_black_kinglike}")
    
    if (game.white_king_pos == expected_white_king and 
        game.white_kinglike_pos == expected_white_kinglike and
        game.black_king_pos == expected_black_king and
        game.black_kinglike_pos == expected_black_kinglike):
        print("\n✓ SUCCESS: All pieces moved correctly!")
    else:
        print("\n✗ FAILURE: Pieces did not move as expected!")
        print("This suggests the server is not processing moves correctly, or our parsing is broken.")

if __name__ == '__main__':
    test_server_moves()

