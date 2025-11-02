import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import multiprocessing
import os
import time
import logging
from enum import Enum
import uuid
import hashlib
from contextlib import contextmanager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenCVのログレベルを設定して、不要な警告を抑制
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore',
                        message=r'.*tf\.lite\.Interpreter is deprecated.*',
                        category=UserWarning)

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


# --- Windowsにおける非ASCIIパスの問題に対するパッチ ---
import sys

if sys.platform == 'win32':
    try:
        import tensorflow as tf
        
        original_interpreter_init = tf.lite.Interpreter.__init__

        def new_interpreter_init(self, model_path=None, model_content=None, **kwargs):
            if model_path and not model_content:
                try:
                    model_path.encode('ascii')
                except UnicodeEncodeError:
                    try:
                        with open(model_path, 'rb') as f:
                            model_content = f.read()
                        model_path = None
                        logger.info("非ASCIIパスを検出: モデルをメモリに読み込みました")
                    except Exception as e:
                        logger.warning(f"モデルの読み込みに失敗: {e}")
            original_interpreter_init(self, model_path=model_path, model_content=model_content, **kwargs)

        tf.lite.Interpreter.__init__ = new_interpreter_init
    except (ImportError, AttributeError) as e:
        logger.debug(f"TensorFlowパッチをスキップ: {e}")

    try:
        import cv2
        import tempfile

        _original_CascadeClassifier = cv2.CascadeClassifier

        class CascadeClassifierWrapper:
            def __init__(self, filename=None):
                self._temp_file = None
                self._classifier = None
                if filename and isinstance(filename, str):
                    try:
                        filename.encode('ascii')
                        self._classifier = _original_CascadeClassifier(filename)
                    except UnicodeEncodeError:
                        try:
                            with open(filename, 'rb') as f:
                                content = f.read()
                            
                            fd, temp_path = tempfile.mkstemp(suffix=".xml")
                            with os.fdopen(fd, 'wb') as temp_f:
                                temp_f.write(content)
                            
                            self._temp_file = temp_path
                            self._classifier = _original_CascadeClassifier(self._temp_file)
                            logger.info("非ASCIIパスを検出: カスケードファイルを一時ファイルにコピーしました")
                        except Exception as e:
                            logger.warning(f"カスケードファイルの読み込みに失敗: {e}")
                            self._classifier = _original_CascadeClassifier(filename)
                elif filename is None:
                    self._classifier = _original_CascadeClassifier()
                else:
                    self._classifier = _original_CascadeClassifier(filename)

            def __getattr__(self, name):
                return getattr(self._classifier, name)

            def __del__(self):
                if self._temp_file:
                    try:
                        os.remove(self._temp_file)
                    except (OSError, AttributeError) as e:
                        logger.debug(f"一時ファイルの削除に失敗: {e}")
        
        cv2.CascadeClassifier = CascadeClassifierWrapper
    except (ImportError, AttributeError) as e:
        logger.debug(f"OpenCVパッチをスキップ: {e}")


@contextmanager
def temporary_audio_file():
    """一時音声ファイルを安全に管理するコンテキストマネージャー"""
    temp_file = os.path.join(TEMP_DIR, f"temp_audio_{uuid.uuid4()}.mp3")
    try:
        yield temp_file
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.debug(f"一時ファイルを削除: {temp_file}")
            except Exception as e:
                logger.warning(f"一時ファイルの削除に失敗: {e}")


def get_video_hash(video_path):
    """動画ファイルのハッシュ値を計算（キャッシュキーとして使用）"""
    try:
        # ファイルサイズが大きい場合は最初と最後の1MBのみ読み込んで高速化
        file_size = os.path.getsize(video_path)
        hash_md5 = hashlib.md5()
        
        with open(video_path, 'rb') as f:
            if file_size < 2 * 1024 * 1024:  # 2MB未満
                # 小さいファイルは全体をハッシュ化
                hash_md5.update(f.read())
            else:
                # 大きいファイルは最初と最後の1MBのみ
                hash_md5.update(f.read(1024 * 1024))
                f.seek(-1024 * 1024, 2)  # 最後から1MB
                hash_md5.update(f.read(1024 * 1024))
        
        return hash_md5.hexdigest()[:16]  # 最初の16文字のみ使用
    except Exception as e:
        logger.error(f"動画ハッシュの計算に失敗: {e}")
        return None


def get_cached_audio_path(video_path):
    """キャッシュされた音声ファイルのパスを取得"""
    video_hash = get_video_hash(video_path)
    if video_hash:
        return os.path.join(TEMP_DIR, f"audio_cache_{video_hash}.mp3")
    return None


def is_cache_valid(video_path, cache_path):
    """キャッシュが有効かどうかをチェック"""
    if not os.path.exists(cache_path):
        return False
    
    try:
        # 動画ファイルの更新日時とキャッシュの更新日時を比較
        video_mtime = os.path.getmtime(video_path)
        cache_mtime = os.path.getmtime(cache_path)
        
        # キャッシュの方が新しい、または同じ時刻なら有効
        return cache_mtime >= video_mtime
    except Exception as e:
        logger.warning(f"キャッシュの有効性チェックに失敗: {e}")
        return False


def video_player_process(video_path, game_over_event, video_ready_event, fullscreen=False,
                         frame_skip_threshold=100, 
                         sync_tolerance_frames=2, 
                         max_frame_wait_ms=50):
    """
    PygameとOpenCVを使用して別のプロセスで動画を再生します。
    """
    import pygame
    import cv2
    import traceback
    from moviepy.editor import VideoFileClip
    
    # Pygame初期化
    pygame.init()
    
    cap = None
    video_fps = 30
    video_size = (640, 480)
    audio_file = None  # 使用する音声ファイル
    total_duration = None # 動画の長さを初期化
    has_audio = False # 音声の有無を初期化

    try:
        print("[動画処理] 動画ファイルをチェック中...")
        if not os.path.exists(video_path):
            logger.error(f"動画ファイルが見つかりません: {video_path}")
            game_over_event.set()
            return

        # --- 音声キャッシュのチェック ---
        print("[動画処理] 音声キャッシュをチェック中...")
        cached_audio_path = get_cached_audio_path(video_path)
        use_cached_audio = False
        
        if cached_audio_path and is_cache_valid(video_path, cached_audio_path):
            print("[動画処理] キャッシュが見つかりました！音声抽出をスキップします")
            logger.info(f"音声キャッシュを使用: {cached_audio_path}")
            audio_file = cached_audio_path
            use_cached_audio = True
            has_audio = True  # 重要: キャッシュがあれば確実に True
            print(f"[動画処理] has_audio を True に設定 (キャッシュ使用)")
        else:
            if cached_audio_path and os.path.exists(cached_audio_path):
                print("[動画処理] キャッシュが古いため、音声を再抽出します")
            else:
                print("[動画処理] キャッシュが見つかりません。音声を抽出します")

        # --- 動画情報の取得と音声抽出 ---
        print("[動画処理] 動画情報を取得中...")
        try:
            with VideoFileClip(video_path) as clip:
                # 動画の基本情報を取得
                video_fps = clip.fps if clip.fps else 30
                video_size = clip.size if clip.size else (640, 480)
                total_duration = clip.duration
                
                # キャッシュがない場合のみ音声を抽出
                if not use_cached_audio:
                    if clip.audio:
                        print("[動画処理] 音声を抽出中...")
                        # キャッシュパスに保存
                        if cached_audio_path:
                            clip.audio.write_audiofile(
                                cached_audio_path, codec='mp3', logger=None)
                            audio_file = cached_audio_path
                            print("[動画処理] 音声抽出完了（キャッシュに保存しました）")
                            logger.info(f"音声をキャッシュに保存: {cached_audio_path}")
                        else:
                            # ハッシュ計算に失敗した場合は一時ファイルを使用
                            temp_audio = os.path.join(TEMP_DIR, f"temp_audio_{os.getpid()}.mp3")
                            clip.audio.write_audiofile(
                                temp_audio, codec='mp3', logger=None)
                            audio_file = temp_audio
                            print("[動画処理] 音声抽出完了")
                        has_audio = True
                        print(f"[動画処理] has_audio を True に設定 (新規抽出)")
                    else:
                        # 音声がない動画
                        print("[動画処理] この動画には音声がありません")
                        has_audio = False
                else:
                    # キャッシュを使用する場合（has_audio は既に True）
                    print(f"[動画処理] キャッシュから音声を使用します (has_audio={has_audio})")
                    logger.info("キャッシュされた音声を使用")
            
            logger.info(f"動画情報取得成功: FPS={video_fps}, サイズ={video_size}, 音声={has_audio}")
            print(f"[動画処理] 動画情報: {video_fps}fps, {video_size[0]}x{video_size[1]}, 長さ: {total_duration:.1f}秒, 音声: {has_audio}")
        except Exception as e:
            logger.warning(f"Moviepyでの処理に失敗: {e}")
            print(f"[動画処理] 警告: Moviepyでの処理に失敗しました: {e}")
            total_duration = None # 失敗した場合はNoneに戻す

        # --- OpenCVでのビデオキャプチャ準備 ---
        print("[動画処理] 動画ファイルを開いています...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"OpenCVで動画ファイルを開けませんでした: {video_path}")
            print("[動画処理] エラー: 動画ファイルを開けませんでした")
            game_over_event.set()
            return

        # OpenCVから情報を取得（Moviepyが失敗した場合のフォールバック）
        if video_fps == 0:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if video_size[0] == 0:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_size = (width, height) if width > 0 else (640, 480)
        if total_duration is None:
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_duration = frame_count / video_fps if video_fps > 0 else 0

        print(f"[動画処理] 動画ファイルを開きました: {video_fps}fps, {video_size[0]}x{video_size[1]}")

        # --- Pygameのセットアップ ---
        print("[動画処理] Pygameを初期化中...")
        pygame.display.set_caption("笑ってはいけないチャレンジ - 動画")
        
        if fullscreen:
            # フルスクリーンモード
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            screen_size = screen.get_size()
            logger.info(f"フルスクリーンモードで起動: {screen_size}")
            
            # アスペクト比を保持してスケーリング
            video_aspect = video_size[0] / video_size[1]
            screen_aspect = screen_size[0] / screen_size[1]
            
            if video_aspect > screen_aspect:
                # 動画の方が横長
                scaled_width = screen_size[0]
                scaled_height = int(screen_size[0] / video_aspect)
            else:
                # 動画の方が縦長
                scaled_height = screen_size[1]
                scaled_width = int(screen_size[1] * video_aspect)
            
            display_size = (scaled_width, scaled_height)
            offset_x = (screen_size[0] - scaled_width) // 2
            offset_y = (screen_size[1] - scaled_height) // 2
            is_fullscreen = True
        else:
            # ウィンドウモード
            screen = pygame.display.set_mode(video_size)
            display_size = video_size
            offset_x = 0
            offset_y = 0
            is_fullscreen = False
            logger.info(f"ウィンドウモードで起動: {video_size}")
        
        clock = pygame.time.Clock()

        # --- 音声再生のセットアップ ---
        audio_ready = False
        if has_audio and audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
            try:
                print("[動画処理] 音声を初期化中...")
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                audio_ready = True
                print("[動画処理] 音声再生を開始しました")
                logger.info("音声再生を開始しました")
            except Exception as e:
                logger.warning(f"音声の初期化に失敗: {e}")
                print(f"[動画処理] 警告: 音声の初期化に失敗しました: {e}")

        # 動画準備完了を通知
        print("[動画処理] ===== 準備完了！再生を開始します =====")
        video_ready_event.set()
        
        # --- 再生ループ ---
        running = True
        playback_start_time = time.time()
        audio_start_time = None
        video_ended = False
        audio_ended = False
        last_sync_log_time = playback_start_time
        frame_duration = 1.0 / video_fps if video_fps > 0 else 0.033

        while running:
            loop_start_time = time.time()
            
            if game_over_event.is_set():
                logger.info("ゲームオーバーイベントを検出")
                break

            # --- 時間同期（音声を主軸とする）---
            if audio_ready and pygame.mixer.music.get_busy():
                if audio_start_time is None:
                    audio_start_time = time.time()
                    logger.info("音声再生を開始しました")
                
                # 音声の再生位置を基準とする（最も信頼できる時刻源）
                audio_pos_ms = pygame.mixer.music.get_pos()
                if audio_pos_ms >= 0:
                    current_time = audio_pos_ms / 1000.0
                else:
                    # get_pos()が負の値を返す場合のフォールバック
                    current_time = time.time() - audio_start_time
            else:
                if audio_ready and not audio_ended:
                    audio_ended = True
                    logger.info("音声の再生が終了しました")
                # 音声がない場合はシステム時刻を使用
                current_time = time.time() - playback_start_time
            
            # デバッグ用：5秒ごとに同期状況をログ
            if time.time() - last_sync_log_time > 5.0:
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                expected_frame = int(current_time * video_fps)
                frame_diff = abs(current_frame - expected_frame)
                logger.debug(
                    f"同期状況 - 時刻: {current_time:.2f}s / {total_duration:.2f}s, "
                    f"フレーム: {current_frame} / {expected_frame} (差: {frame_diff})"
                )
                last_sync_log_time = time.time()

            # --- 動画終了チェック ---
            if total_duration and current_time >= total_duration:
                logger.info("動画の再生時間が終了しました")
                break

            # --- フレーム同期とスキップ ---
            target_frame_num = int(current_time * video_fps)
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_diff = target_frame_num - current_frame_num

            if frame_diff > sync_tolerance_frames:
                # 遅れている場合：フレームをスキップ
                if frame_diff > frame_skip_threshold:
                    # 大きな遅れの場合は直接シーク
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                    logger.debug(f"大幅な遅延を検出: {frame_diff}フレームシーク")
                else:
                    # 小さな遅れの場合はフレームを読み飛ばす
                    skip_count = 0
                    while current_frame_num < target_frame_num - 1:  # 1フレーム前まで
                        ret = cap.grab()
                        if not ret:
                            video_ended = True
                            break
                        current_frame_num += 1
                        skip_count += 1
                    if skip_count > 0:
                        logger.debug(f"{skip_count}フレームをスキップ (遅延: {frame_diff})")
            elif frame_diff < -sync_tolerance_frames:
                # 進みすぎている場合：少し待機
                wait_time = abs(frame_diff) * frame_duration
                if wait_time > 0 and wait_time < max_frame_wait_ms / 1000.0:
                    time.sleep(wait_time)
                    logger.debug(f"進みすぎを検出: {wait_time*1000:.1f}ms待機")

            if video_ended:
                logger.info("動画フレームが終了しました")
                break

            # --- フレーム読み込み ---
            ret, frame = cap.read()
            if not ret:
                logger.info("動画の読み込みが終了しました")
                break

            # --- フレーム表示 ---
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if is_fullscreen:
                    frame_resized = cv2.resize(frame_rgb, display_size)
                    surf = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
                    screen.fill((0, 0, 0))
                    screen.blit(surf, (offset_x, offset_y))
                else:
                    surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    screen.blit(surf, (0, 0))
                
                pygame.display.flip()
            except Exception as e:
                logger.error(f"フレーム描画エラー: {e}")

            # ウィンドウイベントの処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game_over_event.set()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        logger.info("ESCキーが押されました。ゲームを終了します")
                        running = False
                        game_over_event.set()
                    elif event.key == pygame.K_F11 or event.key == pygame.K_f:
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                            screen_size = screen.get_size()
                            video_aspect = video_size[0] / video_size[1]
                            screen_aspect = screen_size[0] / screen_size[1]
                            if video_aspect > screen_aspect:
                                scaled_width = screen_size[0]
                                scaled_height = int(screen_size[0] / video_aspect)
                            else:
                                scaled_height = screen_size[1]
                                scaled_width = int(screen_size[1] * video_aspect)
                            display_size = (scaled_width, scaled_height)
                            offset_x = (screen_size[0] - scaled_width) // 2
                            offset_y = (screen_size[1] - scaled_height) // 2
                            logger.info("フルスクリーンモードに切り替え")
                        else:
                            screen = pygame.display.set_mode(video_size)
                            display_size = video_size
                            offset_x = 0
                            offset_y = 0
                            logger.info("ウィンドウモードに切り替え")

            # フレームレートに基づいた待機時間の計算
            elapsed = time.time() - loop_start_time
            target_frame_time = frame_duration
            sleep_time = target_frame_time - elapsed
            
            if sleep_time > 0.001:  # 1ms以上の場合のみスリープ
                time.sleep(sleep_time)
            elif sleep_time < -0.010:  # 10ms以上遅れている場合は警告
                logger.debug(f"フレーム処理遅延: {-sleep_time*1000:.1f}ms")

        logger.info("動画再生ループを終了します")

    except Exception as e:
        logger.error(f"動画プロセスで予期せぬエラーが発生: {e}")
        logger.error(traceback.format_exc())
        game_over_event.set()
    finally:
        if cap:
            cap.release()
        pygame.quit()
        
        # キャッシュされていない一時ファイルのみ削除
        if audio_file and not (cached_audio_path and audio_file == cached_audio_path):
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logger.debug(f"一時音声ファイルを削除: {audio_file}")
                except Exception as e:
                    logger.warning(f"一時音声ファイルの削除に失敗: {e}")
        
        logger.info("動画再生プロセスが終了しました")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("笑ってはいけないチャレンジ")
        self.geometry("800x700")

        # --- 設定値 (Tkinter変数) ---
        self.smile_threshold = tk.DoubleVar(value=0.70)
        self.frame_resize_scale = tk.DoubleVar(value=0.5)
        self.game_over_duration = tk.DoubleVar(value=5.0)
        self.win_duration = tk.DoubleVar(value=5.0)
        self.camera_search_range = tk.IntVar(value=10)
        self.webcam_update_interval = tk.IntVar(value=15)
        self.video_process_timeout = tk.DoubleVar(value=2.0)
        
        # video_player_process に渡す設定
        self.frame_skip_threshold = tk.IntVar(value=100)
        self.sync_tolerance_frames = tk.IntVar(value=2)
        self.max_frame_wait_ms = tk.IntVar(value=50)
        
        # 設定ウィンドウ管理
        self.settings_window = None

        # 状態管理
        self.game_state = GameState.IDLE
        self.state_change_time = None
        self.win_start_time = None  # 初期化を追加
        self.video_ready_received = False  # 動画準備完了フラグ
        
        # リソース管理
        self.video_path = None
        self.game_over_event = None
        self.video_ready_event = None  # 動画準備完了イベント
        self.video_process = None
        self.detector = None
        self.cap_webcam = None
        self.camera_list = []
        self.selected_camera = tk.StringVar()
        
        # 表示モード設定
        self.display_mode = tk.StringVar(value="window")  # デフォルトはウィンドウモード
        
        # フィード更新制御
        self._is_updating_feed = False
        self._feed_update_id = None

        # --- GUIウィジェット ---
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # --- 上部パネル ---
        self.top_panel = tk.Frame(self.main_frame)
        self.top_panel.pack(fill="x", pady=5)

        # 動画選択
        self.video_frame = tk.Frame(self.top_panel)
        self.video_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.video_label = tk.Label(self.video_frame, text="動画が選択されていません")
        self.video_label.pack(side="left", padx=5)
        self.select_video_button = tk.Button(
            self.video_frame, text="動画を選択", command=self.select_video)
        self.select_video_button.pack(side="right")

        # カメラ選択
        self.webcam_frame = tk.Frame(self.top_panel)
        self.webcam_frame.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self.webcam_label = tk.Label(self.webcam_frame, text="Webカメラ:")
        self.webcam_label.pack(side="left", padx=5)
        self.camera_menu = tk.OptionMenu(
            self.webcam_frame, self.selected_camera, "")
        self.camera_menu.pack(side="left")
        self.refresh_button = tk.Button(
            self.webcam_frame, text="更新", command=self.find_and_update_cameras)
        self.refresh_button.pack(side="left", padx=5)

        # Webカメラ表示
        self.webcam_canvas = tk.Canvas(self.main_frame, bg="black")
        self.webcam_canvas.pack(pady=10, expand=True, fill="both")

        # 操作パネル
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill="x")
        
        # 表示モード選択
        self.display_mode_frame = tk.Frame(self.control_frame)
        self.display_mode_frame.pack(side="left", padx=5)
        tk.Label(self.display_mode_frame, text="表示モード:").pack(side="left")
        self.window_radio = tk.Radiobutton(
            self.display_mode_frame, text="ウィンドウ", 
            variable=self.display_mode, value="window"
        )
        self.window_radio.pack(side="left", padx=2)
        self.fullscreen_radio = tk.Radiobutton(
            self.display_mode_frame, text="フルスクリーン", 
            variable=self.display_mode, value="fullscreen"
        )
        self.fullscreen_radio.pack(side="left", padx=2)
        
        self.start_button = tk.Button(
            self.control_frame, text="ゲーム開始", command=self.start_game, state="disabled")
        self.start_button.pack(side="left", padx=5)

        self.settings_button = tk.Button(
            self.control_frame, text="設定", command=self.open_settings_window)
        self.settings_button.pack(side="left", padx=5)
        
        self.status_label = tk.Label(
            self.control_frame, text="ようこそ！動画とWebカメラを選択してください。")
        self.status_label.pack(side="left", padx=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ESCキーでゲーム終了
        self.bind("<Escape>", self.on_escape_key)

        self.init_face_detector()
        self.selected_camera.trace_add("write", self.on_camera_select)
        self.find_and_update_cameras()

    def init_face_detector(self):
        """顔検出器の初期化"""
        try:
            self.status_label.config(text="顔検出器を初期化しています (mtcnn)...")
            self.update()
            from fer.fer import FER
            self.detector = FER(mtcnn=True)
            logger.info("FER検出器をMTCNNで初期化しました")
        except Exception as e:
            logger.warning(f"MTCNNの初期化に失敗: {e}")
            try:
                self.status_label.config(text="mtcnnなしで再試行しています...")
                self.update()
                from fer.fer import FER
                self.detector = FER(mtcnn=False)
                logger.info("FER検出器をMTCNNなしで初期化しました")
            except Exception as e2:
                logger.error(f"顔検出器の初期化に完全に失敗: {e2}")
                messagebox.showerror("エラー", f"顔検出器の初期化に失敗しました: {e2}")
                return
        self.status_label.config(text="顔検出器の準備ができました。")

    def find_and_update_cameras(self):
        """利用可能なカメラを検索"""
        self.status_label.config(text="利用可能なWebカメラを検索中...")
        self.update()

        if self.cap_webcam and self.cap_webcam.isOpened():
            self.cap_webcam.release()
            self.cap_webcam = None

        self.camera_list = []
        for i in range(self.camera_search_range.get()):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.camera_list.append(f"カメラ {i}")
                    logger.info(f"カメラ {i} を検出しました")
                cap.release()
            except Exception as e:
                logger.debug(f"カメラ {i} のチェックでエラー: {e}")

        menu = self.camera_menu["menu"]
        menu.delete(0, "end")

        if self.camera_list:
            for cam in self.camera_list:
                menu.add_command(
                    label=cam, command=lambda value=cam: self.selected_camera.set(value))
            self.selected_camera.set(self.camera_list[0])
            self.status_label.config(text="Webカメラを選択してください。")
            logger.info(f"{len(self.camera_list)}台のカメラを検出しました")
        else:
            self.selected_camera.set("")
            messagebox.showerror("エラー", "利用可能なWebカメラが見つかりませんでした。")
            self.status_label.config(text="エラー: Webカメラが見つかりません。")
            logger.warning("利用可能なカメラが見つかりませんでした")
        
        self.check_start_button_state()

    def on_camera_select(self, *args):
        """カメラ選択時のハンドラ"""
        selection = self.selected_camera.get()
        if not selection:
            return

        try:
            camera_index = int(selection.split(" ")[1])
            self.initialize_capture(camera_index)
        except (ValueError, IndexError) as e:
            logger.error(f"カメラインデックスの解析に失敗: {e}")

    def initialize_capture(self, camera_index):
        """カメラキャプチャの初期化"""
        # 既存のフィード更新をキャンセル
        if self._feed_update_id:
            self.after_cancel(self._feed_update_id)
            self._feed_update_id = None
        self._is_updating_feed = False

        if self.cap_webcam and self.cap_webcam.isOpened():
            self.cap_webcam.release()

        try:
            import cv2
            self.cap_webcam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            if self.cap_webcam.isOpened():
                self.status_label.config(text=f"Webカメラ {camera_index} を使用中。")
                logger.info(f"カメラ {camera_index} を初期化しました")
                if not self._is_updating_feed:
                    self._is_updating_feed = True
                    self.update_webcam_feed()
            else:
                raise RuntimeError(f"カメラ {camera_index} を開けませんでした")
        except Exception as e:
            logger.error(f"カメラの初期化に失敗: {e}")
            messagebox.showerror("エラー", f"Webカメラ {camera_index} を開けませんでした。")
            self.status_label.config(text=f"エラー: Webカメラ {camera_index} を開けません。")
            self.cap_webcam = None
        
        self.check_start_button_state()

    def check_start_button_state(self):
        """スタートボタンの有効/無効を制御"""
        if self.video_path and self.cap_webcam and self.cap_webcam.isOpened():
            self.start_button.config(state="normal")
        else:
            self.start_button.config(state="disabled")

    def select_video(self):
        """動画ファイルの選択"""
        path = filedialog.askopenfilename(
            title="動画ファイルを選択",
            filetypes=(("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*"))
        )
        if path:
            self.video_path = path
            self.video_label.config(text=os.path.basename(path))
            self.status_label.config(text="動画が選択されました。")
            logger.info(f"動画を選択: {path}")
            self.check_start_button_state()

    def check_video_ready(self):
        """動画の準備完了を監視"""
        if self.video_ready_event and self.video_ready_event.is_set():
            if not self.video_ready_received:
                self.video_ready_received = True
                self.game_state = GameState.PLAYING
                self.status_label.config(text="ゲーム進行中... 笑わないで！")
                logger.info("動画の再生が開始されました。表情判定を開始します。")
                print("[メイン] 動画再生開始を検出。表情判定を開始します。")
        elif self.game_state == GameState.PREPARING and self.video_process and self.video_process.is_alive():
            # まだ準備中の場合は100ms後に再チェック
            self.after(100, self.check_video_ready)

    def start_game(self):
        """ゲームを開始"""
        if not self.video_path:
            messagebox.showwarning("警告", "最初に動画ファイルを選択してください。")
            return
        if not self.cap_webcam or not self.cap_webcam.isOpened():
            messagebox.showwarning("警告", "使用可能なWebカメラが選択されていません。")
            return

        self.start_button.config(state="disabled")
        self.select_video_button.config(state="disabled")
        self.camera_menu.config(state="disabled")
        self.refresh_button.config(state="disabled")
        self.window_radio.config(state="disabled")
        self.fullscreen_radio.config(state="disabled")
        self.status_label.config(text="ゲームを開始しています...")

        self.game_state = GameState.PREPARING
        self.state_change_time = None
        self.video_ready_received = False

        # 表示モードを取得
        fullscreen = (self.display_mode.get() == "fullscreen")
        
        # video_player_process に渡す設定を取得
        process_settings = {
            "frame_skip_threshold": self.frame_skip_threshold.get(),
            "sync_tolerance_frames": self.sync_tolerance_frames.get(),
            "max_frame_wait_ms": self.max_frame_wait_ms.get()
        }
        
        logger.info(f"ゲームを開始しました (表示モード: {'フルスクリーン' if fullscreen else 'ウィンドウ'})")
        logger.info(f"プロセス設定: {process_settings}")

        # --- ★★★ 修正箇所 ★★★ ---
        # 重複していたプロセス起動を1つにまとめます
        
        self.game_over_event = multiprocessing.Event()
        self.video_ready_event = multiprocessing.Event()
        self.video_process = multiprocessing.Process(
            target=video_player_process,
            args=(self.video_path, self.game_over_event, self.video_ready_event, fullscreen),
            kwargs=process_settings # kwargsとして渡す
        )
        self.video_process.start()
        self.status_label.config(text="動画を準備中...")
        
        # 2回目の起動処理を削除
        # --- ここから削除 ---
        # self.game_over_event = multiprocessing.Event()
        # ... (重複ブロック) ...
        # self.video_process.start()
        # self.status_label.config(text="動画を準備中...")
        # logger.info(f"ゲームを開始しました (表示モード: {'フルスクリーン' if fullscreen else 'ウィンドウ'})")
        # --- ここまで削除 ---
        
        # 動画準備完了を監視
        self.check_video_ready()

    def update_webcam_feed(self):
        """Webカメラフィードの更新"""
        if not self._is_updating_feed:
            return

        if not self.cap_webcam or not self.cap_webcam.isOpened():
            self._is_updating_feed = False
            return

        try:
            import cv2
            from PIL import Image, ImageTk
            
            ret, frame = self.cap_webcam.read()
            if not ret:
                logger.warning("Webカメラからフレームを取得できません")
                self._feed_update_id = self.after(100, self.update_webcam_feed)
                return

            # --- ゲームロジック ---
            if self.video_process and self.video_process.is_alive():
                # 動画が実際に再生中で、かつPLAYING状態の場合のみ表情判定を行う
                if self.game_state == GameState.PLAYING and self.video_ready_received:
                    try:
                        resize_scale = self.frame_resize_scale.get()
                        small_frame = cv2.resize(
                            frame, (0, 0),
                            fx=resize_scale,
                            fy=resize_scale
                        )
                        results = self.detector.detect_emotions(small_frame)

                        scale_factor = 1.0 / resize_scale
                        for result in results:
                            x, y, w, h = [int(v * scale_factor) for v in result['box']]
                            emotions = result['emotions']
                            smile_score = emotions.get('happy', 0)

                            if smile_score > self.smile_threshold.get():
                                label = f"笑顔！ ({smile_score:.2f})"
                                color = (0, 255, 0)
                                self.game_state = GameState.GAME_OVER
                                self.state_change_time = time.time()
                                self.game_over_event.set()
                                logger.info(f"笑顔を検出: スコア={smile_score:.2f}")
                            else:
                                dominant_emotion = max(emotions, key=emotions.get)
                                label = dominant_emotion
                                color = (0, 0, 255)

                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                            if self.game_state == GameState.GAME_OVER:
                                break
                    except Exception as e:
                        logger.error(f"感情検出エラー: {e}")
                elif self.game_state == GameState.PREPARING:
                    # 準備中はメッセージを表示
                    cv2.putText(frame, "Preparing...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # --- 状態遷移チェック ---
            if self.game_state == GameState.GAME_OVER:
                out_text = "OUT!"
                text_size = cv2.getTextSize(
                    out_text, cv2.FONT_HERSHEY_TRIPLEX, 5, 10)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, out_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_TRIPLEX, 5, (0, 0, 255), 10)
                self.status_label.config(text="笑いましたね！ゲームオーバーです。")
                
                if self.state_change_time and (time.time() - self.state_change_time > self.game_over_duration.get()):
                    self.reset_game_state()
            
            # ABORTED状態の場合は何も表示せずにリセット待ち
            elif self.game_state == GameState.ABORTED:
                # メッセージボックスが表示されるまで待機
                pass
            
            # 動画終了チェック（PLAYING状態でのみ）
            elif self.video_process and not self.video_process.is_alive() and self.game_state == GameState.PLAYING:
                # 動画が正常に終了し、かつゲーム中の場合のみWINにする
                self.game_state = GameState.WIN
                self.win_start_time = time.time()
                logger.info("ユーザーが勝利しました")

            elif self.game_state == GameState.WIN:
                win_text = "YOU WIN!"
                text_size = cv2.getTextSize(
                    win_text, cv2.FONT_HERSHEY_TRIPLEX, 3, 5)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, win_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0), 5)
                self.status_label.config(text="おめでとうございます！あなたの勝ちです！")
                
                if self.win_start_time and (time.time() - self.win_start_time > self.win_duration.get()):
                    self.reset_game_state()

            # --- フレーム表示 ---
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            canvas_w = self.webcam_canvas.winfo_width()
            canvas_h = self.webcam_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(image=img)
            self.webcam_canvas.create_image(
                canvas_w/2, canvas_h/2, image=self.photo, anchor="center")

        except Exception as e:
            logger.error(f"Webカメラフィード更新エラー: {e}")

        self._feed_update_id = self.after(self.webcam_update_interval.get(), self.update_webcam_feed)

    def reset_game_state(self):
        """ゲーム状態をリセット"""
        logger.info("ゲーム状態をリセットします")
        self.status_label.config(
            text="ゲーム終了。もう一度プレイするには動画を選択してください。")
        self.start_button.config(state="normal")
        self.select_video_button.config(state="normal")
        self.camera_menu.config(state="normal")
        self.refresh_button.config(state="normal")
        self.window_radio.config(state="normal")
        self.fullscreen_radio.config(state="normal")
        
        if self.video_process:
            if self.video_process.is_alive():
                logger.info("動画プロセスを終了します")
                self.video_process.terminate()
                self.video_process.join(timeout=self.video_process_timeout.get())
                if self.video_process.is_alive():
                    logger.warning("動画プロセスが応答しません。強制終了します")
                    self.video_process.kill()
                    self.video_process.join(timeout=1.0)
            self.video_process = None
        
        self.game_state = GameState.IDLE
        self.state_change_time = None
        self.win_start_time = None

    def on_escape_key(self, event=None):
        """ESCキーが押された時の処理"""
        if self.game_state in [GameState.PREPARING, GameState.PLAYING, GameState.GAME_OVER, GameState.WIN]:
            logger.info("ESCキーでゲームを中断します")
            
            # 状態をABORTEDに設定
            self.game_state = GameState.ABORTED
            
            # ゲームオーバーイベントを設定して動画を停止
            if self.game_over_event:
                self.game_over_event.set()
            
            # メッセージボックスを表示
            self.after(100, lambda: [
                messagebox.showinfo("ゲーム中断", "ゲームを中断しました。"),
                self.reset_game_state()
            ])

    def on_closing(self):
        """アプリケーション終了時の処理"""
        if messagebox.askokcancel("終了", "終了しますか？"):
            logger.info("アプリケーションを終了します")
            
            # 設定ウィンドウが開いていれば閉じる
            self.on_close_settings_window()

            # フィード更新を停止
            if self._feed_update_id:
                self.after_cancel(self._feed_update_id)
                self._feed_update_id = None
            self._is_updating_feed = False
            
            # ゲームオーバーイベントを設定
            if self.game_over_event:
                self.game_over_event.set()
            
            # 動画プロセスを終了
            if self.video_process:
                if self.video_process.is_alive():
                    logger.info("動画プロセスを終了しています...")
                    self.video_process.terminate()
                    self.video_process.join(timeout=self.video_process_timeout.get())
                    
                    if self.video_process.is_alive():
                        logger.warning("動画プロセスが応答しません。強制終了します")
                        self.video_process.kill()
                        self.video_process.join(timeout=1.0)
                        
                        if self.video_process.is_alive():
                            logger.error("動画プロセスを終了できませんでした")
            
            # Webカメラを解放
            if self.cap_webcam:
                logger.info("Webカメラを解放します")
                self.cap_webcam.release()
            
            self.destroy()
            logger.info("アプリケーションが正常に終了しました")

    def open_settings_window(self):
        """設定ウィンドウを開く"""
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return

        self.settings_window = tk.Toplevel(self)
        self.settings_window.title("設定")
        self.settings_window.geometry("400x550")
        
        # ウィンドウを閉じたときの処理
        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_close_settings_window)

        main_frame = tk.Frame(self.settings_window)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        tk.Label(main_frame, text="ゲーム設定", font=("", 14, "bold")).pack(pady=5)

        # スライダーの作成 (tk.Scale)
        tk.Scale(main_frame, from_=0.1, to=1.0, resolution=0.01, 
                 orient="horizontal", label="笑顔のしきい値 (SMILE_THRESHOLD)", 
                 variable=self.smile_threshold, length=350).pack(fill="x", pady=5)
                 
        tk.Scale(main_frame, from_=0.1, to=1.0, resolution=0.1, 
                 orient="horizontal", label="カメラリサイズ倍率 (FRAME_RESIZE_SCALE)", 
                 variable=self.frame_resize_scale, length=350).pack(fill="x", pady=5)

        tk.Scale(main_frame, from_=1.0, to=10.0, resolution=0.5, 
                 orient="horizontal", label="ゲームオーバー表示時間(秒)", 
                 variable=self.game_over_duration, length=350).pack(fill="x", pady=5)

        tk.Scale(main_frame, from_=1.0, to=10.0, resolution=0.5, 
                 orient="horizontal", label="勝利表示時間(秒) (WIN_DURATION)", 
                 variable=self.win_duration, length=350).pack(fill="x", pady=5)

        tk.Label(main_frame, text="詳細設定", font=("", 12, "bold")).pack(pady=(10, 5))

        # エントリーの作成 (tk.Entry)
        def create_entry_row(parent, text, variable, from_, to_):
            frame = tk.Frame(parent)
            frame.pack(fill="x", pady=2)
            tk.Label(frame, text=f"{text} ({from_}～{to_}):", width=30, anchor="w").pack(side="left")
            tk.Entry(frame, textvariable=variable, width=10).pack(side="left", padx=5)

        create_entry_row(main_frame, "カメラ検索数", self.camera_search_range, 1, 20)
        create_entry_row(main_frame, "カメラ更新間隔(ms)", self.webcam_update_interval, 10, 100)
        create_entry_row(main_frame, "プロセス終了待機(秒)", self.video_process_timeout, 1.0, 5.0)
        create_entry_row(main_frame, "フレームスキップ閾値", self.frame_skip_threshold, 10, 500)
        create_entry_row(main_frame, "同期許容フレーム", self.sync_tolerance_frames, 1, 10)
        create_entry_row(main_frame, "最大フレーム待機(ms)", self.max_frame_wait_ms, 10, 200)

        tk.Button(main_frame, text="閉じる", command=self.on_close_settings_window).pack(pady=10)

    def on_close_settings_window(self):
        """設定ウィンドウを閉じる"""
        if self.settings_window:
            self.settings_window.destroy()
            self.settings_window = None

def run():
    """アプリケーションのエントリーポイント"""
    multiprocessing.freeze_support()
    logger.info("アプリケーションを起動します")
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        logger.critical(f"致命的なエラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    run()