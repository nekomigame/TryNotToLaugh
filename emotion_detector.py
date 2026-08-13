import os
import sys

# PyTorch関連モデルのキャッシュ・保存先をカレントディレクトリ配下の .cache ディレクトリに設定
_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache"))
os.environ["TORCH_HOME"] = _cache_dir
os.environ["HF_HOME"] = os.path.join(_cache_dir, "huggingface")

import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

EMOTIONS: List[str] = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class PyTorchEmotionClassifier(nn.Module):
    """PyTorchによる感情分類ネットワーク"""
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PyTorchEmotionDetector:
    """
    FERライブラリの代替となるPyTorchベースの顔検出・感情分析クラス
    """
    def __init__(self, mtcnn: bool = True) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mtcnn = mtcnn
        self.mtcnn_detector = None

        # 1. 顔検出器の初期化
        if self.use_mtcnn:
            try:
                from facenet_pytorch import MTCNN
                self.mtcnn_detector = MTCNN(keep_all=True, device=self.device)
                logger.info(f"facenet-pytorch MTCNNを初期化しました (デバイス: {self.device})")
            except Exception as e:
                logger.warning(f"MTCNNの初期化に失敗しました: {e}。OpenCV CascadeClassifierにフォールバックします。")
                self.use_mtcnn = False

        # OpenCV CascadeClassifier (フォールバック用含め常に準備)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("OpenCV CascadeClassifierで顔検出器を初期化しました")
        except Exception as e:
            logger.warning(f"顔カスケードの読み込みに失敗: {e}")
            self.face_cascade = None

        # 笑顔検出器補助
        try:
            smile_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            self.smile_cascade = cv2.CascadeClassifier(smile_path)
        except Exception as e:
            logger.debug(f"笑顔カスケードの読み込みをスキップ: {e}")
            self.smile_cascade = None

        # 2. 感情分類モデルの初期化
        self.model = PyTorchEmotionClassifier(num_classes=len(EMOTIONS))
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """顔画像をグレースケール・48x48に変換しPyTorch Tensorにする"""
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 48, 48]
        return tensor.to(self.device)

    def _detect_faces_opencv(self, img_bgr: np.ndarray) -> List[List[int]]:
        """OpenCV CascadeClassifier を使った顔検出"""
        if self.face_cascade is None:
            return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return []
        return [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]

    def _detect_faces_mtcnn(self, img_bgr: np.ndarray) -> List[List[int]]:
        """MTCNN を使った顔検出"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn_detector.detect(img_rgb)
        if boxes is None:
            return []
        
        face_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w > 10 and h > 10:
                face_boxes.append([x1, y1, w, h])
        return face_boxes

    def detect_emotions(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        画像から顔を検出し、感情スコアのリストを返す
        FERライブラリ互換の出力フォーマット:
        [
            {
                "box": [x, y, width, height],
                "emotions": {
                    "angry": float, "disgust": float, "fear": float,
                    "happy": float, "sad": float, "surprise": float, "neutral": float
                }
            },
            ...
        ]
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        # 顔領域の検出
        boxes = []
        if self.use_mtcnn and self.mtcnn_detector is not None:
            try:
                boxes = self._detect_faces_mtcnn(img_bgr)
            except Exception as e:
                logger.warning(f"MTCNN検出中にエラー発生: {e}。OpenCVに切り替えます。")
                boxes = self._detect_faces_opencv(img_bgr)
        else:
            boxes = self._detect_faces_opencv(img_bgr)

        results = []
        img_h, img_w = img_bgr.shape[:2]

        for box in boxes:
            x, y, w, h = box
            # 範囲外アクセス防止
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            face_roi = img_bgr[y1:y2, x1:x2]

            # 1. PyTorchモデルによる感情推論
            with torch.no_grad():
                face_tensor = self._preprocess_face(face_roi)
                logits = self.model(face_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            emotions_dict = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probs)}

            # 2. 笑顔カスケード補助判定（口元領域の笑顔チェック）
            if self.smile_cascade is not None:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                # 顔の下半分（口元付近）をクロップ
                lower_face = gray_face[int(face_roi.shape[0] * 0.5):, :]
                smiles = self.smile_cascade.detectMultiScale(
                    lower_face, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25)
                )
                if len(smiles) > 0:
                    # 笑顔が検出された場合は happy スコアをブースト
                    emotions_dict['happy'] = max(emotions_dict['happy'], 0.85)

            results.append({
                "box": [x, y, w, h],
                "emotions": emotions_dict
            })

        return results
