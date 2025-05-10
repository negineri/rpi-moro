"""Tests for CameraStreamer SocketIO functionality."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from moro.modules.camera.camera import Camera
from moro.modules.camera.streaming import CameraStreamer


class TestCameraStreamerSocketIO(unittest.TestCase):
    """Test case for the CameraStreamer SocketIO functionality."""

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
    def test_socketio_handlers(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test SocketIO event handlers setup and functionality."""
        # モックの準備
        mock_socketio = mock_socketio_class.return_value

        # CameraStreamerを初期化
        streamer = CameraStreamer(camera=self.mock_camera)

        # キャッシュされたフレームとカメラ情報を設定
        streamer._cached_frame = "test_frame_data"
        streamer._camera_info = {"width": 640, "height": 480, "fps": 30}

        # connect イベントハンドラを取得
        connect_handler = None
        for call in mock_socketio.on.call_args_list:
            args, _ = call
            if args[0] == "connect":
                connect_handler = args[1]
                break

        assert connect_handler is not None, "connect イベントハンドラが登録されていません"

        # connect イベントハンドラを実行
        connect_handler()

        # キャッシュされたフレームとカメラ情報が送信されたか確認
        mock_socketio.emit.assert_any_call("frame", {"frame": "test_frame_data"})
        mock_socketio.emit.assert_any_call("camera_info", streamer._camera_info)
