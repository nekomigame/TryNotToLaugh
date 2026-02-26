import os
from enum import Enum

# --- 定数定義 (変更されないもの) ---
TEMP_DIR = "./temp"
AUDIO_SYNC_OFFSET = 0.1  # 音声同期の補正オフセット（秒） (現在は未使用)

# 一時ディレクトリの作成
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


class GameState(Enum):
    """ゲーム状態を管理するEnum"""
    IDLE = "idle"
    PREPARING = "preparing"  # 動画準備中
    PLAYING = "playing"
    GAME_OVER = "game_over"
    WIN = "win"
    ABORTED = "aborted"
