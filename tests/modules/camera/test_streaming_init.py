"""Tests for CameraStreamer initialization and setup."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from flask import Flask

from moro.modules.camera.camera import Camera
from moro.modules.camera.streaming import CameraStreamer


class TestCameraStreamerInit(unittest.TestCase):
    """Test case for the CameraStreamer initialization and setup."""

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
    def test_camera_streamer_init(
        self, mock_socketio_class: MagicMock, mock_flask_class: MagicMock
    ) -> None:
        """Test CameraStreamer initialization."""
        # Flaskとモックの準備
        mock_flask = mock_flask_class.return_value
        mock_socketio = mock_socketio_class.return_value

        # CameraStreamerのインスタンス化
        streamer = CameraStreamer(
            camera=self.mock_camera,
            fps=15,
            quality=70,
            host="127.0.0.1",
            port=5000,
            resolution=(320, 240),
            grayscale=False,
            motion_detection=False,
        )

        # 引数が正しく設定されているか確認
        assert streamer.camera == self.mock_camera
        assert streamer.fps == 15
        assert streamer.quality == 70
        assert streamer.host == "127.0.0.1"
        assert streamer.port == 5000
        assert streamer.resolution == (320, 240)
        assert streamer.grayscale is False
        assert streamer.motion_detection is False
        assert streamer.stream_active is False

        # Flaskアプリが適切に作成されたか確認
        mock_flask_class.assert_called_once()
        assert streamer.app == mock_flask
        mock_flask.register_blueprint.assert_called_once()

        # SocketIOが適切に初期化されたか確認
        mock_socketio_class.assert_called_once()
        assert streamer.socketio == mock_socketio

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_setup_routes(
        self, mock_socketio_class: MagicMock, mock_flask_class: MagicMock
    ) -> None:
        """Test route setup."""
        # Flaskのテストクライアントを作成
        test_app = Flask(__name__)
        mock_flask_class.return_value = test_app

        # CameraStreamerを初期化
        streamer = CameraStreamer(
            camera=self.mock_camera,
            fps=15,
            quality=70,
        )

        # ルートが登録されていることを確認
        assert "camera" in streamer.app.blueprints
