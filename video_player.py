import os
import time
from utils import logger, get_cached_audio_path, is_cache_valid
from constants import TEMP_DIR

def video_player_process(video_path, game_over_event, video_ready_event, win_event, fullscreen=False,
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
    import tempfile 
    import shutil 
    
    # Pygame初期化
    pygame.init()
    
    cap = None
    video_fps = 30
    video_size = (640, 480)
    audio_file = None  # 使用する音声ファイル
    total_duration = None # 動画の長さを初期化
    has_audio = False # 音声の有無を初期化
    temp_video_file = None # 非ASCIIパス用

    try:
        # --- 非ASCIIパス対応 ---
        try:
            video_path.encode('ascii')
            logger.debug("動画パスはASCIIです。")
            video_path_to_use = video_path
        except UnicodeEncodeError:
            logger.info("非ASCII動画パスを検出しました。一時ファイルにコピーします。")
            try:
                # 拡張子を保持した一時ファイルを作成
                ext = os.path.splitext(video_path)[1]
                fd, temp_video_file = tempfile.mkstemp(suffix=ext, dir=TEMP_DIR)
                os.close(fd) # ファイルディスクリプタを閉じる
                
                shutil.copy(video_path, temp_video_file)
                video_path_to_use = temp_video_file
                logger.info(f"一時ファイルにコピー完了: {temp_video_file}")
            except Exception as e:
                logger.warning(f"一時ファイルへの動画コピーに失敗: {e}。元のパスを試行します。")
                video_path_to_use = video_path
        
        print("[動画処理] 動画ファイルをチェック中...")
        if not os.path.exists(video_path_to_use):
            logger.error(f"動画ファイルが見つかりません: {video_path_to_use}")
            game_over_event.set()
            return

        # --- 音声キャッシュのチェック ---
        # (キャッシュキーは元のパス `video_path` を使用)
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
            with VideoFileClip(video_path_to_use) as clip:
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
        cap = cv2.VideoCapture(video_path_to_use)
        if not cap.isOpened():
            logger.error(f"OpenCVで動画ファイルを開けませんでした: {video_path_to_use}")
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

        # game_over_eventがセットされていなければ、正常終了とみなしwin_eventをセット
        if not game_over_event.is_set():
            logger.info("動画が正常に終了しました。勝利イベントをセットします。")
            win_event.set()

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
        
        # --- 一時動画ファイルの削除 ---
        if temp_video_file and os.path.exists(temp_video_file):
            try:
                os.remove(temp_video_file)
                logger.info(f"一時動画ファイルを削除: {temp_video_file}")
            except Exception as e:
                logger.warning(f"一時動画ファイルの削除に失敗: {e}")

        logger.info("動画再生プロセスが終了しました")
