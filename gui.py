import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import multiprocessing
import os
import time

# OpenCVのログレベルを設定して、不要な警告を抑制
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
# --- 警告を抑制 ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore',
                        message=r'.*tf\.lite\.Interpreter is deprecated.*',
                        category=UserWarning)

# --- Windowsにおける非ASCIIパスの問題に対するパッチ ---
import sys

if sys.platform == 'win32':
    # TensorFlow Lite や OpenCV などのライブラリが、
    # Windows 上で非ASCII文字を含むパスからファイルを読み込めない問題を解決。

    # 1. TensorFlow Lite モデルの読み込みに関するパッチ
    try:
        import tensorflow as tf
        
        original_interpreter_init = tf.lite.Interpreter.__init__

        def new_interpreter_init(self, model_path=None, model_content=None, **kwargs):
            if model_path and not model_content:
                try:
                    # パスがASCII文字のみで構成されている場合は、元のローダーに処理を任せる。
                    model_path.encode('ascii')
                except UnicodeEncodeError:
                    # パスに非ASCII文字が含まれている場合は、その内容をそのまま渡す。
                    try:
                        with open(model_path, 'rb') as f:
                            model_content = f.read()
                        model_path = None # パスではなくコンテンツを使用する
                    except Exception:
                        # 失敗時は元の動作に戻す
                        pass
            original_interpreter_init(self, model_path=model_path, model_content=model_content, **kwargs)

        tf.lite.Interpreter.__init__ = new_interpreter_init
    except (ImportError, AttributeError):
        # TensorFlowがインストールされていない、またはアクセスできない場合は無視
        pass

    # 2. OpenCVのCascadeClassifier読み込みに関するパッチ
    try:
        import cv2
        import tempfile

        _original_CascadeClassifier = cv2.CascadeClassifier

        class CascadeClassifierWrapper:
            """
            非ASCIIパスを処理するためのcv2.CascadeClassifierのラッパー。
            カスケードファイルを読み込むために、ASCIIパスに一時ファイルを作成。
            """
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
                        except Exception:
                            # 失敗時は元の動作に戻す
                            self._classifier = _original_CascadeClassifier(filename)
                elif filename is None:
                    self._classifier = _original_CascadeClassifier()
                else:
                    self._classifier = _original_CascadeClassifier(filename)


            def __getattr__(self, name):
                # すべてのメソッド/属性呼び出しをラップされたオブジェクトに転送する
                return getattr(self._classifier, name)

            def __del__(self):
                # オブジェクトが破棄されるときに一時ファイルを削除する
                if self._temp_file:
                    try:
                        os.remove(self._temp_file)
                    except (OSError, AttributeError):
                        pass
        
        cv2.CascadeClassifier = CascadeClassifierWrapper
    except (ImportError, AttributeError):
        # OpenCVがインストールされていない、またはアクセスできない場合は無視
        pass
# --- パッチ終了 ---

from moviepy.editor import VideoFileClip
import pygame
from fer.fer import FER
from PIL import Image, ImageTk
import cv2


# --- デフォルト設定 ---
SMILE_THRESHOLD = 0.70

# このディレクトリが存在しない場合は作成されます。
if not os.path.exists("./temp"):
    os.makedirs("./temp")


def video_player_process(video_path, game_over_event):
    """
    PygameとOpenCVを使用して別のプロセスで動画を再生します。
    Moviepyは音声抽出にのみ使用します。
    """
    temp_audio_file = f"./temp/temp_audio_{os.getpid()}.mp3"
    cap = None
    video_fps = 30  # デフォルト値
    video_size = (640, 480)  # デフォルト値

    try:
        if not os.path.exists(video_path):
            print(f"動画ファイルが見つかりません: {video_path}")
            game_over_event.set()
            return

        # --- 音声抽出 (Moviepy) ---
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio:
                    clip.audio.write_audiofile(
                        temp_audio_file, codec='mp3', logger=None)
                video_fps = clip.fps
                video_size = clip.size
        except Exception as e:
            print(f"Moviepyでの動画情報取得または音声抽出に失敗: {e}")
            # Moviepyが失敗しても、OpenCVで試行を続ける

        # --- OpenCVでのビデオキャプチャ準備 ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"OpenCVで動画ファイルを開けませんでした: {video_path}")
            game_over_event.set()
            return

        # Moviepyがプロパティを取得できなかった場合、OpenCVから取得
        if video_fps is None or video_fps == 0:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_size is None or video_size[0] == 0:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_size = (width, height)

        # --- Pygameのセットアップ ---
        pygame.display.set_caption("笑ってはいけないチャレンジ - 動画")
        screen = pygame.display.set_mode(video_size)
        clock = pygame.time.Clock()

        # --- 音声再生のセットアップ ---
        has_audio = os.path.exists(
            temp_audio_file) and os.path.getsize(temp_audio_file) > 0
        if has_audio:
            pygame.mixer.init()
            pygame.mixer.music.load(temp_audio_file)
            pygame.mixer.music.play()

        # --- 再生ループ ---
        running = True
        start_time = time.time()

        while running:
            if game_over_event.is_set():
                break

            # --- 時間同期 ---
            if has_audio and pygame.mixer.music.get_busy():
                current_time = pygame.mixer.music.get_pos() / 1000.0
            else:
                if has_audio:  # 音声がちょうど終了した
                    break
                current_time = time.time() - start_time

            # --- フレームのスキップ/読み込み ---
            target_frame_num = int(current_time * video_fps)

            # OpenCVのフレーム位置設定は正確でない場合があるため、手動で同期
            # 現在のフレーム番号を取得
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # ターゲットフレームに追いつくまでフレームを読み飛ばす
            if current_frame_num < target_frame_num:
                # 差が大きすぎる場合は直接シークする（パフォーマンスのため）
                if target_frame_num - current_frame_num > 100:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                else:
                    while current_frame_num < target_frame_num:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        current_frame_num += 1

            ret, frame = cap.read()
            if not ret:
                break  # 動画の終端

            # --- フレーム表示 ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # ウィンドウイベントの処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game_over_event.set()

            # FPSを維持しようと試みるが、同期が優先
            clock.tick(video_fps * 2)  # 同期のため少し高めに設定

    except Exception as e:
        import traceback
        print(f"動画プロセスで予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        game_over_event.set()
    finally:
        if cap:
            cap.release()
        pygame.quit()
        if os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except Exception as e:
                print(f"一時音声ファイルの削除に失敗しました: {e}")
        print("動画再生プロセスが終了しました。")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("笑ってはいけないチャレンジ")
        self.geometry("800x700")

        self.video_path = None
        self.game_over_event = None
        self.video_process = None
        self.detector = None
        self.cap_webcam = None
        self.game_over = False
        self.game_over_time = None
        self.user_wins = False
        self.camera_list = []
        self.selected_camera = tk.StringVar()
        self._is_updating_feed = False

        # --- GUIウィジェット ---
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # --- 上部パネル (動画とカメラ選択) ---
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
        self.webcam_frame.pack(side="left", fill="x",
                               expand=True, padx=(10, 0))
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
        self.start_button = tk.Button(
            self.control_frame, text="ゲーム開始", command=self.start_game, state="disabled")
        self.start_button.pack(side="left", padx=5)
        self.status_label = tk.Label(
            self.control_frame, text="ようこそ！動画とWebカメラを選択してください。")
        self.status_label.pack(side="left", padx=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.init_face_detector()

        # カメラのセットアップ
        self.selected_camera.trace_add("write", self.on_camera_select)
        self.find_and_update_cameras()

    def init_face_detector(self):
        try:
            self.status_label.config(
                text="顔検出器を初期化しています (mtcnn)...")
            self.update()
            self.detector = FER(mtcnn=True)
        except Exception as e:
            self.status_label.config(
                text=f"mtcnnの初期化に失敗: {e}。mtcnnなしで再試行します。")
            self.update()
            self.detector = FER(mtcnn=False)
        self.status_label.config(text="顔検出器の準備ができました。")

    def find_and_update_cameras(self):
        self.status_label.config(
            text="利用可能なWebカメラを検索中...検索中はUIがフリーズする可能性があります。")
        self.update()

        if self.cap_webcam and self.cap_webcam.isOpened():
            self.cap_webcam.release()
            self.cap_webcam = None

        self.camera_list = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_list.append(f"カメラ {i}")
                cap.release()

        menu = self.camera_menu["menu"]
        menu.delete(0, "end")

        if self.camera_list:
            for cam in self.camera_list:
                menu.add_command(
                    label=cam, command=lambda value=cam: self.selected_camera.set(value))
            self.selected_camera.set(self.camera_list[0])
            self.status_label.config(text="Webカメラを選択してください。")
        else:
            self.selected_camera.set("")
            messagebox.showerror("エラー", "利用可能なWebカメラが見つかりませんでした。")
            self.status_label.config(text="エラー: Webカメラが見つかりません。")
        self.check_start_button_state()

    def on_camera_select(self, *args):
        selection = self.selected_camera.get()
        if not selection:
            return

        camera_index = int(selection.split(" ")[1])
        self.initialize_capture(camera_index)

    def initialize_capture(self, camera_index):
        if self.cap_webcam and self.cap_webcam.isOpened():
            self.cap_webcam.release()

        self.cap_webcam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if self.cap_webcam.isOpened():
            self.status_label.config(text=f"Webカメラ {camera_index} を使用中。")
            if not self._is_updating_feed:
                self._is_updating_feed = True
                self.update_webcam_feed()
        else:
            messagebox.showerror("エラー", f"Webカメラ {camera_index} を開けませんでした。")
            self.status_label.config(
                text=f"エラー: Webカメラ {camera_index} を開けません。")
            self.cap_webcam = None
        self.check_start_button_state()

    def check_start_button_state(self):
        if self.video_path and self.cap_webcam and self.cap_webcam.isOpened():
            self.start_button.config(state="normal")
        else:
            self.start_button.config(state="disabled")

    def select_video(self):
        path = filedialog.askopenfilename(
            title="動画ファイルを選択",
            filetypes=(("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*"))
        )
        if path:
            self.video_path = path
            self.video_label.config(text=os.path.basename(path))
            self.status_label.config(text="動画が選択されました。")
            self.check_start_button_state()

    def start_game(self):
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
        self.status_label.config(text="ゲームを開始しています...")

        self.game_over = False
        self.user_wins = False
        self.game_over_time = None

        self.game_over_event = multiprocessing.Event()
        self.video_process = multiprocessing.Process(
            target=video_player_process, args=(
                self.video_path, self.game_over_event)
        )
        self.video_process.start()
        self.status_label.config(text="ゲーム進行中... 笑わないで！")

    def update_webcam_feed(self):
        if not self.cap_webcam or not self.cap_webcam.isOpened():
            self._is_updating_feed = False
            return

        ret, frame = self.cap_webcam.read()
        if not ret:
            self.status_label.config(
                text="エラー: Webカメラからフレームを取得できません。")
            self.after(100, self.update_webcam_feed)
            return

        # --- ゲームロジック ---
        if self.video_process and self.video_process.is_alive():
            if not self.game_over:
                try:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    results = self.detector.detect_emotions(small_frame)

                    for result in results:
                        x, y, w, h = [v * 2 for v in result['box']]
                        emotions = result['emotions']
                        smile_score = emotions.get('happy', 0)

                        if smile_score > SMILE_THRESHOLD:
                            label = f"笑顔！ ({smile_score:.2f})"
                            color = (0, 255, 0)
                            self.game_over = True
                            self.game_over_time = time.time()
                            self.game_over_event.set()
                        else:
                            dominant_emotion = max(emotions, key=emotions.get)
                            label = dominant_emotion
                            color = (0, 0, 255)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        if self.game_over:
                            break
                except Exception:
                    pass

        # --- ゲームオーバー/勝利表示 ---
        if self.game_over:
            out_text = "OUT!"
            text_size = cv2.getTextSize(
                out_text, cv2.FONT_HERSHEY_TRIPLEX, 5, 10)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, out_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_TRIPLEX, 5, (0, 0, 255), 10)
            self.status_label.config(text="笑いましたね！ゲームオーバーです。")
            if self.game_over_time and (time.time() - self.game_over_time > 5):
                self.reset_game_state()

        if self.video_process and not self.video_process.is_alive() and not self.game_over and not self.user_wins:
            self.user_wins = True
            self.win_start_time = time.time()

        if self.user_wins:
            win_text = "YOU WIN!"
            text_size = cv2.getTextSize(
                win_text, cv2.FONT_HERSHEY_TRIPLEX, 3, 5)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, win_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0), 5)
            self.status_label.config(text="おめでとうございます！あなたの勝ちです！")
            if time.time() - self.win_start_time > 5:
                self.reset_game_state()

        # --- Tkinterキャンバスにフレームを表示 ---
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        canvas_w = self.webcam_canvas.winfo_width()
        canvas_h = self.webcam_canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image=img)
        self.webcam_canvas.create_image(
            canvas_w/2, canvas_h/2, image=self.photo, anchor="center")

        self.after(15, self.update_webcam_feed)

    def reset_game_state(self):
        self.status_label.config(
            text="ゲーム終了。もう一度プレイするには動画を選択してください。")
        self.start_button.config(state="normal")
        self.select_video_button.config(state="normal")
        self.camera_menu.config(state="normal")
        self.refresh_button.config(state="normal")
        if self.video_process:
            if self.video_process.is_alive():
                self.video_process.terminate()
            self.video_process.join()
            self.video_process = None
        self.game_over = False
        self.user_wins = False
        self.game_over_time = None

    def on_closing(self):
        if messagebox.askokcancel("終了", "終了しますか？"):
            if self.game_over_event:
                self.game_over_event.set()
            if self.video_process:
                if self.video_process.is_alive():
                    self.video_process.terminate()
                self.video_process.join(timeout=1.0)
            if self.cap_webcam:
                self.cap_webcam.release()
            self.destroy()


def run():
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()


if __name__ == '__main__':
    run()
