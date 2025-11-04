import urllib.request
import zipfile
import os
import io
import sys

# Gyan.devから提供されているWindows向けFFmpeg essentialsビルドのURL
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def download_and_extract():
    """FFmpegをダウンロードして、必要な実行ファイルをカレントディレクトリに展開する"""
    target_dir = "."
    print(f"FFmpegをダウンロードしています...\nURL: {FFMPEG_URL}")

    try:
        # メモリ内でzipファイルを扱うためにBytesIOを使用
        with urllib.request.urlopen(FFMPEG_URL) as response:
            if response.status != 200:
                print(
                    f"エラー: ファイルのダウンロードに失敗しました (ステータスコード: {response.status})", file=sys.stderr)
                return

            # ダウンロードサイズが大きい可能性を考慮し、進捗を表示
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            bytes_read = 0

            zip_content = io.BytesIO()
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
                zip_content.write(chunk)
                if total_size > 0:
                    progress = (bytes_read / total_size) * 100
                    sys.stdout.write(
                        f"\rダウンロード中: {bytes_read / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB ({progress:.1f}%) ")
                    sys.stdout.flush()
            print("\nダウンロード完了。")

        print("zipファイルを展開しています...")
        zip_content.seek(0)
        with zipfile.ZipFile(zip_content) as zf:
            # zipファイル内のbinフォルダにある実行ファイルのみを対象とする
            # 例: ffmpeg-6.0-essentials_build/bin/ffmpeg.exe -> ffmpeg.exe
            for member in zf.infolist():
                parts = member.filename.split('/')
                # `bin`フォルダの直下にあるファイルかチェック
                if len(parts) > 1 and parts[-2] == 'bin':
                    # 展開先のファイル名をルートディレクトリ直下になるように調整
                    target_filename = os.path.basename(member.filename)
                    if target_filename:  # ディレクトリ自体はスキップ
                        print(f"  - {target_filename} を展開中...")
                        # ファイルのデータを読み込む
                        source = zf.open(member)
                        # カレントディレクトリに書き出す
                        target_path = os.path.join(target_dir, target_filename)
                        with open(target_path, "wb") as target_file:
                            target_file.write(source.read())

        print("\n展開完了。ffmpeg.exeがプロジェクトフォルダに配置されました。")
        print("アプリを再起動すると、自動的に認識されます。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}", file=sys.stderr)
        print("インターネット接続を確認するか、手動でFFmpegをダウンロードして配置してください。", file=sys.stderr)


if __name__ == "__main__":
    # スクリプトが直接実行された場合に処理を開始
    download_and_extract()
