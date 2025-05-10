"""Tests for CameraStreamer frame processing functionality."""

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from moro.modules.camera.camera import Camera
from moro.modules.camera.streaming import CameraStreamer


class TestCameraStreamerFrameProcessing(unittest.TestCase):
    """Test case for the CameraStreamer frame processing functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # モックカメラを作成
        self.mock_camera = MagicMock(spec=Camera)
        self.mock_camera.is_running = True
        self.mock_camera.device_id = 0
        self.mock_camera.width = 640
        self.mock_camera.height = 480
        self.mock_camera.fps = 30

        # テスト用のダミーフレームを作成
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_process_frame_resize(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test frame processing with resizing."""
        # CameraStreamerを初期化
        streamer = CameraStreamer(
            camera=self.mock_camera,
            resolution=(320, 240),
            grayscale=False,
            motion_detection=False,
        )

        # cv2.resizeをモック
        with patch("moro.modules.camera.streaming.cv2.resize") as mock_resize:
            mock_resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

            # フレーム処理を実行
            result = streamer._process_frame(self.dummy_frame)

            # リサイズが呼ばれたか確認
            mock_resize.assert_called()
            assert result is not None

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_process_frame_grayscale(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test frame processing with grayscale conversion."""
        # CameraStreamerを初期化
        streamer = CameraStreamer(
            camera=self.mock_camera,
            grayscale=True,
            motion_detection=False,
        )

        # cv2.cvtColorをモック
        with patch("moro.modules.camera.streaming.cv2.cvtColor") as mock_cvtcolor:
            mock_cvtcolor.side_effect = [
                np.zeros((480, 640, 1), dtype=np.uint8),  # グレースケール変換
                np.zeros((480, 640, 3), dtype=np.uint8),  # BGR変換
            ]

            # フレーム処理を実行
            result = streamer._process_frame(self.dummy_frame)

            # grayscale変換が呼ばれたか確認
            assert mock_cvtcolor.call_count == 2
            assert result is not None

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_process_frame_motion_detection(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test frame processing with motion detection."""
        # CameraStreamerを初期化
        streamer = CameraStreamer(
            camera=self.mock_camera,
            motion_detection=True,
            motion_threshold=0.005,
        )

        # 前のフレームを設定
        streamer._previous_frame = self.dummy_frame.copy()

        # モックを設定
        with (
            patch("moro.modules.camera.streaming.cv2.cvtColor") as mock_cvtcolor,
            patch("moro.modules.camera.streaming.cv2.GaussianBlur") as mock_blur,
            patch("moro.modules.camera.streaming.cv2.absdiff") as mock_absdiff,
            patch("moro.modules.camera.streaming.cv2.threshold") as mock_threshold,
            patch("moro.modules.camera.streaming.np.count_nonzero") as mock_count,
        ):
            # モーション検出の過程をモック
            gray_frame = np.zeros((480, 640), dtype=np.uint8)
            mock_cvtcolor.return_value = gray_frame
            mock_blur.return_value = gray_frame
            mock_absdiff.return_value = gray_frame
            mock_threshold.return_value = (None, gray_frame)

            # 変更ピクセル数を設定（閾値以下）
            mock_count.return_value = 10  # 低い値 -> モーションなし

            # フレーム処理を実行
            result = streamer._process_frame(self.dummy_frame)

            # モーション検出で閾値以下なのでNoneが返る
            assert result is None
            assert streamer._skipped_frames_count == 1

            # 閾値以上に設定
            mock_count.return_value = int(gray_frame.size * 0.01)  # 閾値以上 -> モーションあり
            streamer._skipped_frames_count = 5  # 何フレームかスキップした後

            # フレーム処理を実行
            result = streamer._process_frame(self.dummy_frame)

            # モーション検出されたのでフレームが返る
            assert result is not None
            assert streamer._skipped_frames_count == 0

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_process_frame_motion_detection_max_skip(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test frame processing with motion detection at max skip frames."""
        # CameraStreamerを初期化
        streamer = CameraStreamer(
            camera=self.mock_camera,
            motion_detection=True,
            motion_threshold=0.005,
            motion_max_skip_frames=10,
        )

        # 前のフレームを設定
        streamer._previous_frame = self.dummy_frame.copy()
        streamer._skipped_frames_count = 10  # 最大スキップフレーム

        # フレーム処理を実行（最大スキップフレーム数に達した場合）
        result = streamer._process_frame(self.dummy_frame)

        # 最大スキップフレーム数に達したのでリセットしてフレームが返る
        assert result is not None
        assert streamer._skipped_frames_count == 0

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_process_frame_combined_features(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test frame processing with all features combined (resize + grayscale + motion detection)."""
        # CameraStreamerを初期化（すべての機能を有効に）
        streamer = CameraStreamer(
            camera=self.mock_camera,
            resolution=(320, 240),
            grayscale=True,
            motion_detection=True,
            motion_threshold=0.01,
        )

        # 前のフレームを設定
        streamer._previous_frame = cv2.resize(self.dummy_frame.copy(), streamer.resolution)

        # モックを設定
        with (
            patch("moro.modules.camera.streaming.cv2.resize") as mock_resize,
            patch("moro.modules.camera.streaming.cv2.cvtColor") as mock_cvtcolor,
            patch("moro.modules.camera.streaming.cv2.GaussianBlur") as mock_blur,
            patch("moro.modules.camera.streaming.cv2.absdiff") as mock_absdiff,
            patch("moro.modules.camera.streaming.cv2.threshold") as mock_threshold,
            patch("moro.modules.camera.streaming.np.count_nonzero") as mock_count,
        ):
            # リサイズ結果の設定
            resized_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            mock_resize.return_value = resized_frame

            # グレースケール変換の結果を設定
            gray_frame = np.zeros((240, 320), dtype=np.uint8)
            color_frame = np.zeros((240, 320, 3), dtype=np.uint8)

            # モックの戻り値
            mock_cvtcolor.side_effect = [
                gray_frame,  # モーション検出のためのグレースケール変換 (現フレーム)
                gray_frame,  # モーション検出のためのグレースケール変換 (前フレーム)
                gray_frame,  # ユーザーが要求したグレースケール変換
                color_frame,  # グレースケールから再変換
            ]
            mock_blur.return_value = gray_frame
            mock_absdiff.return_value = gray_frame
            mock_threshold.return_value = (None, gray_frame)

            # 閾値以上の変更を検出（モーションあり）
            thresh_size = 320 * 240
            mock_count.return_value = int(thresh_size * 0.02)  # 閾値(0.01)より大きい

            # フレーム処理を実行
            result = streamer._process_frame(self.dummy_frame)

            # リサイズが呼ばれたことを確認
            mock_resize.assert_called()

            # グレースケール変換（モーション検出用）が呼ばれたことを確認
            assert mock_cvtcolor.call_count >= 4

            # モーション検出のための処理が呼ばれたことを確認
            mock_blur.assert_called()
            mock_absdiff.assert_called()
            mock_threshold.assert_called()
            mock_count.assert_called()

            # 結果フレームがNoneでないことを確認（モーション検出されたため）
            assert result is not None

            # 前フレームが更新されたことを確認
            assert streamer._previous_frame is not None

            # スキップフレームカウンターがリセットされたことを確認
            assert streamer._skipped_frames_count == 0

            # シナリオ2: モーションが検出されない場合
            mock_count.return_value = int(thresh_size * 0.005)  # 閾値(0.01)より小さい

            # フレーム処理を実行
            result = streamer._process_frame(self.dummy_frame)

            # モーション検出されないのでNoneが返る
            assert result is None

            # スキップフレームカウンターがインクリメントされたことを確認
            assert streamer._skipped_frames_count == 1
