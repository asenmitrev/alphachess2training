"""
Telegram bot to play King Capture against the trained AlphaZero model.
"""
import os
import logging
import asyncio
import numpy as np
import torch
import glob
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler
from game import KingCapture, Player, Piece
from model import AlphaZeroNet
from mcts import MCTS
from config import Config

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Silence noisy HTTP request logs from python-telegram-bot's HTTP stack.
# (PTB v20+ uses httpx/httpcore internally.)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Silence most telegram library logs (keep warnings/errors).
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)

# Global model and config
model = None
config = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Store active games: chat_id -> game_state
# game_state = {
#     'game': KingCapture(),
#     'mcts': MCTS(...),
#     'user_plays_white': bool,
#     'selected_piece': int or None  # piece_idx if piece selected, None otherwise
# }
games = {}

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load a trained model from checkpoint."""
    global config
    config = Config()
    
    # Initialize model
    model = AlphaZeroNet(
        board_size=config.board_size,
        num_channels=config.num_channels,
        num_residual_blocks=config.num_residual_blocks
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def list_checkpoints(checkpoint_dir: str = 'checkpoints'):
    """List available checkpoints."""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'model_iter_*.pth'))
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoints

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]

def get_board_keyboard(game: KingCapture, user_plays_white: bool, selected_piece: int = None):
    """Create InlineKeyboardMarkup for the game board."""
    keyboard = []
    # Piece symbols
    symbols = {
        Piece.EMPTY.value: '⬜',
        Piece.WHITE_KING.value: '👑',
        Piece.WHITE_KINGLIKE.value: '⚪',
        Piece.BLACK_KING.value: '♔',
        Piece.BLACK_KINGLIKE.value: '⚫'
    }
    
    # Get valid moves for current player
    valid_moves = game.get_valid_moves() if not game.game_over else []
    # If a piece is selected, filter to only moves for that piece
    if selected_piece is not None:
        valid_moves = [m for m in valid_moves if m[0] == selected_piece]
    valid_move_set = set(valid_moves)
    
    # Determine which pieces belong to current player
    current_player = game.current_player
    user_is_current = (current_player == Player.WHITE) == user_plays_white
    
    for row in range(game.BOARD_SIZE):
        row_buttons = []
        for col in range(game.BOARD_SIZE):
            val = game.board[row, col]
            text = symbols[val]
            
            # Check if this is a user's piece (for selection)
            is_user_piece = False
            if user_plays_white:
                is_user_piece = val == Piece.WHITE_KING.value or val == Piece.WHITE_KINGLIKE.value
            else:
                is_user_piece = val == Piece.BLACK_KING.value or val == Piece.BLACK_KINGLIKE.value
            
            # Determine piece_idx if this is a user's piece
            # Check actual piece positions to determine which piece this is
            piece_idx = None
            if is_user_piece:
                if user_plays_white:
                    if game.white_king_pos == (row, col):
                        piece_idx = 0
                    elif game.white_kinglike_pos == (row, col):
                        piece_idx = 1
                else:
                    if game.black_king_pos == (row, col):
                        piece_idx = 0
                    elif game.black_kinglike_pos == (row, col):
                        piece_idx = 1
            
            # Determine callback
            if game.game_over:
                callback_data = "ignore"
            elif selected_piece is not None:
                # Piece selected, show valid moves for that piece
                if (selected_piece, row, col) in valid_move_set:
                    callback_data = f"move_{selected_piece}_{row}_{col}"
                else:
                    callback_data = "ignore"
            elif is_user_piece and user_is_current:
                # User's piece, allow selection
                callback_data = f"select_{piece_idx}"
            elif val == Piece.EMPTY.value:
                # Empty cell, check if it's a valid move destination
                # Find if any piece can move here
                can_move_here = any((p, row, col) in valid_move_set for p in [0, 1])
                if can_move_here and user_is_current:
                    # Show as selectable (will need to select piece first)
                    callback_data = "select_piece"
                else:
                    callback_data = "ignore"
            else:
                callback_data = "ignore"
            
            row_buttons.append(InlineKeyboardButton(text, callback_data=callback_data))
        keyboard.append(row_buttons)
    
    # Add control buttons
    if selected_piece is not None and not game.game_over:
        keyboard.append([InlineKeyboardButton("Cancel Selection", callback_data="cancel_select")])
    
    if game.game_over:
        keyboard.append([InlineKeyboardButton("New Game", callback_data="new_game")])
        
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "Welcome to AlphaChess Bot! 🤖\n\n"
        "I am an AlphaZero-based AI trained to play King Capture.\n"
        "Use /play to start a new game.\n\n"
        "Rules:\n"
        "• Move your pieces (king 👑 or kinglike ⚪) one square in any direction\n"
        "• Capture the opponent's true king to win\n"
        "• Or move your true king to the opponent's starting row\n"
        "• Click your piece to select it, then click a valid destination"
    )

async def play_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /play command."""
    keyboard = [
        [
            InlineKeyboardButton("Play as White (Go First)", callback_data="start_white"),
            InlineKeyboardButton("Play as Black (Go Second)", callback_data="start_black")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose your side:", reply_markup=reply_markup)

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all callback queries."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    chat_id = query.message.chat_id
    
    if data == "ignore":
        return

    if data == "new_game":
         await query.edit_message_text(
            "Choose your side:",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Play as White (Go First)", callback_data="start_white"),
                    InlineKeyboardButton("Play as Black (Go Second)", callback_data="start_black")
                ]
            ])
        )
         return

    if data == "cancel_select":
        if chat_id not in games:
            await query.edit_message_text("Game session expired. Use /play to start a new game.")
            return
        games[chat_id]['selected_piece'] = None
        game_state = games[chat_id]
        game = game_state['game']
        user_plays_white = game_state['user_plays_white']
        await query.edit_message_text("Your turn! Select a piece to move.", 
                                     reply_markup=get_board_keyboard(game, user_plays_white))
        return

    if data.startswith("select_"):
        if chat_id not in games:
            await query.edit_message_text("Game session expired. Use /play to start a new game.")
            return
        
        game_state = games[chat_id]
        game = game_state['game']
        user_plays_white = game_state['user_plays_white']
        
        if game.game_over:
            await query.edit_message_text("Game is over.", reply_markup=get_board_keyboard(game, user_plays_white))
            return
        
        # Check if it's user's turn
        is_white_turn = (game.current_player == Player.WHITE)
        if is_white_turn != user_plays_white:
            await query.answer("It's not your turn!", show_alert=True)
            return
        
        if data == "select_piece":
            await query.answer("Please select a piece first, then the destination.", show_alert=True)
            return
        
        # Extract piece_idx
        piece_idx = int(data.split('_')[1])
        
        # Validate piece exists and belongs to current player
        if user_plays_white:
            if piece_idx == 0 and game.white_king_pos is None:
                await query.answer("This piece has been captured!", show_alert=True)
                return
            if piece_idx == 1 and game.white_kinglike_pos is None:
                await query.answer("This piece has been captured!", show_alert=True)
                return
        else:
            if piece_idx == 0 and game.black_king_pos is None:
                await query.answer("This piece has been captured!", show_alert=True)
                return
            if piece_idx == 1 and game.black_kinglike_pos is None:
                await query.answer("This piece has been captured!", show_alert=True)
                return
        
        game_state['selected_piece'] = piece_idx
        
        await query.edit_message_text("Piece selected! Choose destination.", 
                                     reply_markup=get_board_keyboard(game, user_plays_white, piece_idx))
        return

    if data in ["start_white", "start_black"]:
        user_plays_white = (data == "start_white")
        
        # Initialize game
        game = KingCapture()
        mcts = MCTS(model, num_simulations=400, c_puct=config.c_puct, 
                   device=device)
        
        games[chat_id] = {
            'game': game,
            'mcts': mcts,
            'user_plays_white': user_plays_white,
            'selected_piece': None
        }
        
        if user_plays_white:
            msg = "Game started! You are White 👑⚪. Your turn."
            await query.edit_message_text(msg, reply_markup=get_board_keyboard(game, user_plays_white))
        else:
            msg = "Game started! You are Black ♔⚫. AI (White) is thinking..."
            await query.edit_message_text(msg, reply_markup=get_board_keyboard(game, user_plays_white))
            # AI makes first move
            await make_ai_move(query, chat_id)
            
    elif data.startswith("move_"):
        # User move
        if chat_id not in games:
            await query.edit_message_text("Game session expired. Use /play to start a new game.")
            return

        game_state = games[chat_id]
        game = game_state['game']
        user_plays_white = game_state['user_plays_white']
        
        if game.game_over:
            await query.edit_message_text("Game is over.", reply_markup=get_board_keyboard(game, user_plays_white))
            return

        # Check if it's user's turn
        is_white_turn = (game.current_player == Player.WHITE)
        if is_white_turn != user_plays_white:
            await query.answer("It's not your turn!", show_alert=True)
            return

        # Parse move: move_piece_idx_row_col
        parts = data.split('_')
        piece_idx = int(parts[1])
        row = int(parts[2])
        col = int(parts[3])
        
        if not game.make_move(piece_idx, row, col):
             await query.answer("Invalid move!", show_alert=True)
             return
        
        # Clear selection
        game_state['selected_piece'] = None
             
        # Update board immediately
        await query.edit_message_text("AI is thinking...", reply_markup=get_board_keyboard(game, user_plays_white))
        
        # Check game over
        if game.game_over:
            await handle_game_over(query, game, user_plays_white)
            return
            
        # AI Turn
        await make_ai_move(query, chat_id)

async def make_ai_move(query, chat_id):
    """Execute AI move and update UI."""
    game_state = games[chat_id]
    game = game_state['game']
    mcts = game_state['mcts']
    user_plays_white = game_state['user_plays_white']
    
    # AI thinking...
    policy = mcts.search(game)

    # Choose an AI move by sampling from the MCTS policy (less deterministic than argmax).
    # This avoids "always the same moves" when multiple actions are plausible.
    PLAY_TEMPERATURE = 0.8  # 1.0 = sample as-is, <1 sharper, >1 flatter, 0.0 = greedy
    PLAY_EPSILON = 0.05     # mix in a bit of uniform over valid moves for variety

    valid_moves = game.get_valid_moves()
    valid_actions = [game.move_to_action(piece_idx, row, col) for piece_idx, row, col in valid_moves]

    action = None
    if valid_actions:
        probs = policy[valid_actions].astype(np.float64)

        # Robust renormalization in case policy is slightly off / degenerate.
        probs_sum = float(probs.sum())
        if probs_sum <= 0.0 or not np.isfinite(probs_sum):
            probs = np.ones(len(valid_actions), dtype=np.float64) / float(len(valid_actions))
        else:
            probs = probs / probs_sum

        # Add a bit of uniform exploration.
        if PLAY_EPSILON > 0:
            probs = (1.0 - PLAY_EPSILON) * probs + PLAY_EPSILON * (1.0 / float(len(valid_actions)))

        # Temperature: sample from probs^(1/T). If T==0, fall back to greedy.
        if PLAY_TEMPERATURE and PLAY_TEMPERATURE > 0:
            probs = probs ** (1.0 / float(PLAY_TEMPERATURE))
            probs = probs / float(probs.sum())
            action = int(np.random.choice(valid_actions, p=probs))
        else:
            action = int(valid_actions[int(np.argmax(probs))])

    if action is not None:
        piece_idx, row, col = game.action_to_move(action)
        game.make_move(piece_idx, row, col)
        
    if game.game_over:
        await handle_game_over(query, game, user_plays_white)
    else:
        await query.edit_message_text("Your turn! Select a piece to move.", 
                                     reply_markup=get_board_keyboard(game, user_plays_white))

async def handle_game_over(query, game, user_plays_white):
    """Handle end of game."""
    if game.winner is None:
        msg = "It's a draw!"
    elif (game.winner == Player.WHITE and user_plays_white) or \
         (game.winner == Player.BLACK and not user_plays_white):
        msg = "Congratulations! You won! 🎉"
    else:
        msg = "AI wins! 🤖"
        
    await query.edit_message_text(msg, reply_markup=get_board_keyboard(game, user_plays_white))


def main():
    """Main function."""
    # Get token
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        print("Please export TELEGRAM_BOT_TOKEN='your_token_here'")
        return

    # Load model
    global model
    checkpoint_path = get_latest_checkpoint()
    if not checkpoint_path:
        print("No checkpoints found!")
        return
        
    try:
        model = load_model(checkpoint_path, device=device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Build Application
    application = ApplicationBuilder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("play", play_command))
    application.add_handler(CallbackQueryHandler(handle_callback))

    # Run
    print("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()
