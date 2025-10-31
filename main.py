import multiprocessing
from gui import run

if __name__ == '__main__':
    # Windowsで実行可能ファイルを作成する際に必要
    multiprocessing.freeze_support()
    # GUIアプリケーションを起動
    run()
