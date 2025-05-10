"""Tests for CameraStreamer start and stop functionality."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from moro.modules.camera.camera import Camera
from moro.modules.camera.streaming import CameraStreamer


class TestCameraStreamerStartStop(unittest.TestCase):
    """Test case for the CameraStreamer start and stop functionality."""

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

    @patch("moro.modules.camera.streaming.threading.Thread")
    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_start_camera_not_running(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
        mock_thread_class: MagicMock,
    ) -> None:
        """Test starting streamer with non-running camera."""
        # カメラが実行されていない状態を設定
        self.mock_camera.is_running = False

        # CameraStreamerを初期化
        streamer = CameraStreamer(camera=self.mock_camera)

        # startメソッドをモック
        streamer.socketio.run = MagicMock()

        # ストリーミングを開始
        streamer.start()

        # カメラが開始されたか確認
        self.mock_camera.start.assert_called_once()
        assert streamer.stream_active is True

        # ストリーミングスレッドが作成され、開始されたか確認
        mock_thread_class.assert_called_once()
        mock_thread_instance = mock_thread_class.return_value
        mock_thread_instance.start.assert_called_once()

        # Flaskサーバーが開始されたか確認
        streamer.socketio.run.assert_called_once_with(
            streamer.app, host=streamer.host, port=streamer.port, debug=False
        )

    @patch("moro.modules.camera.streaming.threading.Thread")
    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_start_already_active(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
        mock_thread_class: MagicMock,
    ) -> None:
        """Test starting streamer when it's already active."""
        # CameraStreamerを初期化
        streamer = CameraStreamer(camera=self.mock_camera)
        streamer.stream_active = True

        # startメソッドをモック
        streamer.socketio.run = MagicMock()

        # ストリーミングを開始
        streamer.start()

        # すでにアクティブな場合、何も起こらない
        self.mock_camera.start.assert_not_called()
        mock_thread_class.assert_not_called()
        streamer.socketio.run.assert_not_called()

    @patch("moro.modules.camera.streaming.time.sleep")
    @patch("moro.modules.camera.streaming.threading.Thread")
    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_start_with_preload_frames(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
        mock_thread_class: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Test starting streamer with preloaded frames."""
        # テストフレームを設定
        self.mock_camera.read_frame.return_value = self.dummy_frame

        # CameraStreamerを初期化
        streamer = CameraStreamer(camera=self.mock_camera, preload_frames=True)
        streamer.socketio.run = MagicMock()

        # ストリーミングを開始
        with patch("moro.modules.camera.streaming.cv2.imencode") as mock_imencode:
            # cv2.imencodeをモック
            mock_imencode.return_value = (True, b"dummy_jpeg_data")

            streamer.start()

            # プリロードが行われたか確認
            assert self.mock_camera.read_frame.call_count >= 3
            mock_imencode.assert_called_once()
            assert streamer._cached_frame is not None

    @patch("moro.modules.camera.streaming.threading.Thread")
    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_stop(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
        mock_thread_class: MagicMock,
    ) -> None:
        """Test stopping the streamer."""
        # CameraStreamerを初期化
        streamer = CameraStreamer(camera=self.mock_camera)
        streamer.stream_active = True

        # モックスレッドを設定
        mock_thread = mock_thread_class.return_value
        mock_thread.is_alive.return_value = True
        streamer._stream_thread = mock_thread

        # ストリーミングを停止
        streamer.stop()

        # 停止イベントが設定されたか確認
        assert streamer._stop_event.is_set()

        # スレッドが終了を待っていることを確認
        mock_thread.join.assert_called_once()

        # ストリーミングがアクティブでなくなったか確認
        assert streamer.stream_active is False

    @patch("moro.modules.camera.streaming.threading.Thread")
    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_stop_not_active(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
        mock_thread_class: MagicMock,
    ) -> None:
        """Test stopping the streamer when it's not active."""
        # CameraStreamerを初期化
        streamer = CameraStreamer(camera=self.mock_camera)
        streamer.stream_active = False

        # 停止イベントをモック
        streamer._stop_event.set = MagicMock()

        # ストリーミングを停止
        streamer.stop()

        # アクティブでない場合、何も起こらない
        streamer._stop_event.set.assert_not_called()
