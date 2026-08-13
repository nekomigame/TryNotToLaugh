import os
import logging
import uuid
import hashlib
from contextlib import contextmanager
import shutil
import sys
from constants import TEMP_DIR

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Windowsにおける非ASCIIパスの問題に対するパッチ ---
if sys.platform == 'win32':
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
