import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import multiprocessing
import os
import queue
import threading
import sys
import shutil

# OpenCVのログレベルを設定して、不要な警告を抑制
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

# 抽出したモジュールのインポート
from constants import GameState
from utils import logger, check_and_install_ffmpeg
from video_player import video_player_process

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
        
        # バリデーションコマンドの登録
        self.vcmd_int = (self.register(self._validate_int), '%P')
        self.vcmd_float = (self.register(self._validate_float), '%P')

        # 状態管理
        self.game_state = GameState.IDLE
        self.state_change_time = None
        self.win_start_time = None  # 初期化を追加
        self.video_ready_received = False  # 動画準備完了フラグ
        self.detector_ready = False # GUIフリーズ対策
        self.detector = None

        # 表情認識用スレッド関連
        self.frame_queue = queue.Queue(maxsize=1)
        self.emotion_queue = queue.Queue(maxsize=1)
        self.emotion_thread_active = True
        self.emotion_thread = threading.Thread(target=self._emotion_worker, daemon=True)
        self.emotion_thread.start()
        self.latest_emotions = [] # 最新の認識結果を保持

        # リソース管理
        self.video_path = None
        self.game_over_event = None
        self.video_ready_event = None  # 動画準備完了イベント
        self.win_event = None          # 勝利イベント
        self.video_process = None
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

        self.init_face_detector() # 別スレッドで初期化を開始
        self.selected_camera.trace_add("write", self.on_camera_select)
        self.find_and_update_cameras()

    def _validate_int(self, value_if_allowed):
        """整数入力のバリデーション"""
        if value_if_allowed == "":
            return True  # 空の入力を許可
        try:
            val = int(value_if_allowed)
            return val >= 0 # 0以上のみ許可 (例として)
        except ValueError:
            return False

    def _validate_float(self, value_if_allowed):
        """浮動小数点数入力のバリデーション"""
        if value_if_allowed == "":
            return True  # 空の入力を許可
        try:
            float(value_if_allowed)
            return True
        except ValueError:
            return False

    def init_face_detector(self):
        """顔検出器の初期化 (別スレッドで実行)"""
        self.status_label.config(text="顔検出器を初期化しています...")
        self.detector_ready = False
        self.check_start_button_state() # スタートボタンを無効化

        # キューの初期化
        self.detector_queue = queue.Queue()
        
        # ポーリング開始
        self.check_detector_queue()

        # スレッドを作成して実行
        thread = threading.Thread(target=self._load_detector_thread)
        thread.daemon = True # メインスレッド終了時にスレッドも終了
        thread.start()

    def check_detector_queue(self):
        """検出器初期化スレッドからのメッセージを処理"""
        try:
            while True:
                # ブロックせずにキューから取得
                msg_type, data = self.detector_queue.get_nowait()
                
                if msg_type == "ready":
                    self._on_detector_ready(data)
                    return # 完了したのでポーリング終了
                elif msg_type == "status":
                    self.status_label.config(text=data)
                elif msg_type == "error":
                    self._on_detector_failed(data)
                    return # 失敗したのでポーリング終了
                    
                self.detector_queue.task_done()
        except queue.Empty:
            pass
            
        # ポーリング継続
        self.after(100, self.check_detector_queue)

    def _emotion_worker(self):
        """別スレッドで表情認識を行うワーカー処理"""
        while self.emotion_thread_active:
            try:
                # フレームが来るまで最大0.1秒待機
                frame = self.frame_queue.get(timeout=0.1)
                
                if self.detector is None or not self.detector_ready:
                    self.frame_queue.task_done()
                    continue

                # 検出処理 (時間のかかる処理)
                results = self.detector.detect_emotions(frame)
                
                # 古い結果があれば破棄して最新のみをキューに入れる
                try:
                    while not self.emotion_queue.empty():
                        self.emotion_queue.get_nowait()
                except queue.Empty:
                    pass
                    
                self.emotion_queue.put(results)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"感情検出スレッドエラー: {e}")

    def _load_detector_thread(self):
        """顔検出器をロードするワーカースレッド"""
        try:
            logger.info("PyTorch顔検出器 (MTCNN) の初期化を開始...")
            from emotion_detector import PyTorchEmotionDetector
            detector = PyTorchEmotionDetector(mtcnn=True)
            logger.info("PyTorch感情検出器をMTCNNで初期化しました")
            
            # 完了をキューに送信
            self.detector_queue.put(("ready", detector))
            
        except Exception as e:
            logger.warning(f"MTCNNの初期化に失敗: {e}")
            try:
                logger.info("MTCNNなしで再試行しています...")
                # ステータス更新をキューに送信
                self.detector_queue.put(("status", "MTCNNなしで再試行しています..."))
                
                from emotion_detector import PyTorchEmotionDetector
                detector = PyTorchEmotionDetector(mtcnn=False)
                logger.info("PyTorch感情検出器をMTCNNなしで初期化しました")
                
                # 完了をキューに送信
                self.detector_queue.put(("ready", detector))
                
            except Exception as e2:
                logger.error(f"顔検出器の初期化に完全に失敗: {e2}")
                # エラーをキューに送信
                self.detector_queue.put(("error", e2))

    def _on_detector_ready(self, detector):
        """顔検出器の準備が完了した (メインスレッドで実行)"""
        self.detector = detector
        self.detector_ready = True
        self.status_label.config(text="顔検出器の準備ができました。")
        self.check_start_button_state() # スタートボタンの状態を更新

    def _on_detector_failed(self, error):
        """顔検出器の準備が失敗した (メインスレッドで実行)"""
        self.detector_ready = False
        messagebox.showerror("エラー", f"顔検出器の初期化に失敗しました: {error}")
        self.status_label.config(text="エラー: 顔検出器の初期化に失敗。")
        self.check_start_button_state()

    def find_and_update_cameras(self):
        """利用可能なカメラを検索"""
        import cv2
        self.status_label.config(text="利用可能なWebカメラを検索中...")
        self.update()

        if self.cap_webcam and self.cap_webcam.isOpened():
            self.cap_webcam.release()
            self.cap_webcam = None

        self.camera_list = []
        for i in range(self.camera_search_range.get()):
            try:
                # DirectShow バックエンドを明示的に指定して高速化＆安定化 (Windows向け)
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
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
        import cv2
        # 既存のフィード更新をキャンセル
        if self._feed_update_id:
            self.after_cancel(self._feed_update_id)
            self._feed_update_id = None
        self._is_updating_feed = False

        if self.cap_webcam and self.cap_webcam.isOpened():
            self.cap_webcam.release()

        try:
            # DirectShow バックエンドを明示的に指定して高速化＆安定化 (Windows向け)
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
        if self.video_path and self.cap_webcam and self.cap_webcam.isOpened() and self.detector_ready:
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
        
        # 検出器の準備ができていない場合は開始しない
        if not self.detector_ready:
            messagebox.showwarning("警告", "顔検出器がまだ準備中です。")
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
        
        self.game_over_event = multiprocessing.Event()
        self.video_ready_event = multiprocessing.Event()
        self.win_event = multiprocessing.Event()
        self.video_process = multiprocessing.Process(
            target=video_player_process,
            args=(self.video_path, self.game_over_event, self.video_ready_event, self.win_event, fullscreen),
            kwargs=process_settings # kwargsとして渡す
        )
        self.video_process.start()
        self.status_label.config(text="動画を準備中...")
        
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
            import time
            from PIL import Image, ImageTk
            
            ret, frame = self.cap_webcam.read()
            if not ret:
                logger.warning("Webカメラからフレームを取得できません")
                self._feed_update_id = self.after(100, self.update_webcam_feed)
                return

            # --- ゲームロジック ---
            # self.detector が None でないことも確認
            if self.detector and self.video_process and self.video_process.is_alive():
                # 動画が実際に再生中で、かつPLAYING状態の場合のみ表情判定を行う
                if self.game_state == GameState.PLAYING and self.video_ready_received:
                    try:
                        resize_scale = self.frame_resize_scale.get()
                        small_frame = cv2.resize(
                            frame, (0, 0),
                            fx=resize_scale,
                            fy=resize_scale
                        )
                        
                        # フレームをワーカーに送る (キューが空の場合のみ)
                        if self.frame_queue.empty():
                            self.frame_queue.put(small_frame)
                        
                        # 結果があれば取得する
                        try:
                            while not self.emotion_queue.empty():
                                self.latest_emotions = self.emotion_queue.get_nowait()
                                self.emotion_queue.task_done()
                        except queue.Empty:
                            pass
                            
                        results = self.latest_emotions

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
                # プロセスは終了したが、WINかどうかを判定する
                if self.win_event and self.win_event.is_set():
                    # 正常終了した場合
                    self.game_state = GameState.WIN
                    self.win_start_time = time.time()
                    logger.info("ユーザーが勝利しました")
                else:
                    # 異常終了または中断された場合
                    logger.warning("動画プロセスが異常終了または中断されました。ゲームをリセットします。")
                    self.status_label.config(text="動画の再生が中断されました。")
                    self.reset_game_state()

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

        # --- プロセスに終了イベントを送信 ---
        if self.game_over_event:
            self.game_over_event.set()
        
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
        self.win_event = None

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
            
            # スレッドの終了
            self.emotion_thread_active = False

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
        self.settings_window.geometry("400x600")
        
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
        def create_entry_row(parent, text, variable, from_, to_, validation_cmd):
            frame = tk.Frame(parent)
            frame.pack(fill="x", pady=2)
            tk.Label(frame, text=f"{text} ({from_}～{to_}):", width=30, anchor="w").pack(side="left")
            
            # バリデーションコマンドを追加
            entry = tk.Entry(frame, textvariable=variable, width=10, 
                             validate="key", validatecommand=validation_cmd)
            entry.pack(side="left", padx=5)

        # バリデーションコマンドを渡す
        create_entry_row(main_frame, "カメラ検索数", self.camera_search_range, 1, 20, self.vcmd_int)
        create_entry_row(main_frame, "カメラ更新間隔(ms)", self.webcam_update_interval, 10, 100, self.vcmd_int)
        create_entry_row(main_frame, "プロセス終了待機(秒)", self.video_process_timeout, 1.0, 5.0, self.vcmd_float)
        create_entry_row(main_frame, "フレームスキップ閾値", self.frame_skip_threshold, 10, 500, self.vcmd_int)
        create_entry_row(main_frame, "同期許容フレーム", self.sync_tolerance_frames, 1, 10, self.vcmd_int)
        create_entry_row(main_frame, "最大フレーム待機(ms)", self.max_frame_wait_ms, 10, 200, self.vcmd_int)

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

def check_and_install_ffmpeg():
    """
    FFmpegの存在をチェックし、なければインストールする。
    優先順位:
    1. カレントディレクトリ (ffmpeg.exe)
    2. 環境変数 PATH
    3. 上記で見つからなければダウンロード
    """
    # 1. カレントディレクトリに ffmpeg.exe があるかチェック (Windowsを想定)
    ffmpeg_exe_path = os.path.join(".", "ffmpeg.exe")
    if os.path.exists(ffmpeg_exe_path):
        logger.info(f"カレントディレクトリで ffmpeg.exe を見つけました: {os.path.abspath(ffmpeg_exe_path)}")
        return True

    # 2. 環境変数 PATH 内に ffmpeg が存在するかチェック
    found_path = shutil.which('ffmpeg')
    if found_path:
        logger.info(f"環境変数 PATH 内で FFmpeg を見つけました: {found_path}")
        return True

    # 3. どこにも見つからない場合はダウンロードを試みる
    logger.warning("FFmpeg が見つかりません。ダウンロードを開始します...")
    
    try:
        # install_ffmpeg.py をインポートして実行
        import install_ffmpeg
        install_ffmpeg.download_and_extract()
        
        # ダウンロード後にもう一度カレントディレクトリをチェック
        if os.path.exists(ffmpeg_exe_path):
            logger.info("FFmpegのインストールが完了しました。")
            return True
        else:
            logger.error("FFmpegのダウンロードまたは展開に失敗しました。")
            return False
            
    except ImportError:
        logger.error("install_ffmpeg.py が見つかりません。自動インストールはスキップされます。")
        return False
    except Exception as e:
        logger.error(f"FFmpegのインストール中にエラーが発生しました: {e}")
        return False

if __name__ == '__main__':
    # アプリケーションを実行する前にFFmpegの準備ができているか確認
    if check_and_install_ffmpeg():
    # FFmpegが準備できた場合のみアプリを起動
        run()
    else:
        # 失敗した場合はユーザーに通知して終了
        print("\n--- エラー ---", file=sys.stderr)
        print("FFmpegの準備に失敗したため、アプリケーションを起動できません。", file=sys.stderr)
        print("手動でFFmpegをインストールし、環境変数PATHを通すか、", file=sys.stderr)
        print("もしくは、ffmpeg.exe をこのプログラムと同じフォルダに配置してください。", file=sys.stderr)
        # コンソールが一瞬で閉じないように待機
        input("何かキーを押して終了します...")