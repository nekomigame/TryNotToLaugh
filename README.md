# TryNotToLaugh(笑ってはいけないチャレンジ)
このプログラムでは動画を再生している間、表情検出を用いて笑ったかを判定し笑ったと判定された場合動画が自動で終了します。  
動画は自由に選択することが出来るので好みの動画で試してみてください。  
WEBカメラ必須なのでない場合は用意してください。  

# 動作確認環境
<table>
  <tr>
    <td>os</td>
    <td>CPU</td>
    <td>GPU</td>
    <td>Python Version</td>
  </tr>
  <tr>
    <td>Windows 11</td>
    <td>i7 9700</td>
    <td>GeForce GTX 1660 SUPER</td>
    <td>3.11.3</td>
  </tr>
  <tr>
  <tr>
    <td>Windows 11</td>
    <td>i5 1235U</td>
    <td>Intel Iris Xe Graphics</td>
    <td>3.11.0</td>
  </tr>
</table>

# 使い方  
Pythonが実行できる環境を用意してください。できれば3.10系か3.11系を用意してください。  
gitを用いてcloneするかcodeからDownload ZIPでコードをダウンロードしてください。  
以下のコマンドをPower Shellに入力して実行してください。  
```
cd <ディレクトリパス>
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
python gui.py
```
コードを実行してエラーウィンドウが表示されなければ正常に実行できています。  
あとは好きな動画を選択し、笑ってはいけないチャレンジを開始してください。

# 注意
ライブラリにTensorFlowを用いているため、ファイルパスに日本語（非ASCIIコード）が入っているとエラーが発生する可能性があります。