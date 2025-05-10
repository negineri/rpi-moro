"""Tests for CameraStreamer streaming functionality and camera info handling."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from moro.modules.camera.camera import Camera, CameraError
from moro.modules.camera.streaming import CameraStreamer


class TestCameraStreamerStreaming(unittest.TestCase):
    """Test case for the CameraStreamer core streaming functionality."""

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
    def test_stream_frames_camera_error(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
        mock_thread_class: MagicMock,
    ) -> None:
        """Test stream_frames with camera error."""
        # CameraStreamerを初期化
        streamer = CameraStreamer(camera=self.mock_camera)

        # カメラエラーを設定
        self.mock_camera.read_frame.side_effect = CameraError("Test camera error")

        # ストリーミングスレッドを実行
        streamer._stream_frames()

        # エラーによりストリーミングが停止したことを確認
        assert streamer._stop_event.is_set()

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    @patch("moro.modules.camera.streaming.time.sleep")
    @patch("moro.modules.camera.streaming.time.time")
    @patch("moro.modules.camera.streaming.cv2.imencode")
    @patch("moro.modules.camera.streaming.base64.b64encode")
    def test_stream_frames_normal(
        self,
        mock_b64encode: MagicMock,
        mock_imencode: MagicMock,
        mock_time: MagicMock,
        mock_sleep: MagicMock,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test normal operation of the _stream_frames method."""
        # モックの戻り値を設定
        self.mock_camera.read_frame.return_value = self.dummy_frame
        self.mock_camera.get_camera_info.return_value = {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
        }

        mock_imencode.return_value = (True, b"dummy_jpeg_data")
        mock_b64encode.return_value = b"base64_encoded_frame"

        # time.time() の戻り値を設定
        mock_time.side_effect = [0, 0.05, 10.0, 10.05]  # 初回と2回目、その後10秒経過後

        # CameraStreamerを初期化
        mock_socketio = mock_socketio_class.return_value
        streamer = CameraStreamer(camera=self.mock_camera, fps=20)

        # 停止用のフラグを設定
        streamer._stop_event.set = MagicMock(
            side_effect=lambda: setattr(streamer, "_test_stop_after_frames", True)
        )

        # 2フレーム処理したら停止するようにする
        def stream_frames_with_limit() -> None:
            streamer._test_stop_after_frames = False
            frame_count = 0
            while not streamer._stop_event.is_set() and frame_count < 2:
                streamer._stream_frames_original()
                frame_count += 1
                if frame_count >= 2:
                    streamer._stop_event.set()

        # 元のメソッドを保存
        streamer._stream_frames_original = streamer._stream_frames
        streamer._stream_frames = stream_frames_with_limit

        # ストリーミングスレッドを実行
        streamer._stream_frames()

        # カメラ情報が送信されたか確認
        mock_socketio.emit.assert_any_call(
            "camera_info",
            {
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "stream_fps": 20,
                "stream_quality": 70,
                "stream_resolution": "Original",
                "stream_grayscale": False,
                "stream_motion_detection": False,
                "stream_motion_threshold": "N/A",
                "stream_motion_max_skip": "N/A",
            },
        )

        # フレームが送信されたか確認（少なくとも1回以上）
        frame_calls = [call for call in mock_socketio.emit.call_args_list if call[0][0] == "frame"]
        assert len(frame_calls) > 0, "フレームが送信されていません"

        # フレームデータの形式をチェック
        frame_data = frame_calls[0][0][1]
        assert "frame" in frame_data
        assert frame_data["frame"] == "base64_encoded_frame"

        # キャッシュが更新されたか確認
        assert streamer._cached_frame is not None

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_camera_info_emission(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test camera info retrieval and emission."""
        # モックの準備
        mock_socketio = mock_socketio_class.return_value

        # カメラ情報の設定
        self.mock_camera.get_camera_info.return_value = {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "format": "MJPG",
        }

        # CameraStreamerを初期化（特殊な設定で）
        streamer = CameraStreamer(
            camera=self.mock_camera,
            fps=15,
            quality=80,
            resolution=(320, 240),
            grayscale=True,
            motion_detection=True,
            motion_threshold=0.01,
            motion_max_skip_frames=20,
        )

        # _stream_framesの最初の部分だけを抽出して実行
        with patch.object(streamer, "_stop_event") as mock_stop_event:
            # すぐに停止するようにする
            mock_stop_event.is_set.side_effect = [False, True]

            # カメラが実行中に設定
            self.mock_camera.is_running = True

            # ストリーミングスレッド関数を実行
            streamer._stream_frames()

        # 正しいカメラ情報が送信されたか確認
        expected_camera_info = {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "format": "MJPG",
            "stream_fps": 15,
            "stream_quality": 80,
            "stream_resolution": "(320, 240)",
            "stream_grayscale": True,
            "stream_motion_detection": True,
            "stream_motion_threshold": 0.01,
            "stream_motion_max_skip": 20,
        }

        mock_socketio.emit.assert_any_call("camera_info", expected_camera_info)
        assert streamer._camera_info == expected_camera_info

    @patch("moro.modules.camera.streaming.Flask")
    @patch("moro.modules.camera.streaming.SocketIO")
    def test_camera_info_emission_error(
        self,
        mock_socketio_class: MagicMock,
        mock_flask_class: MagicMock,
    ) -> None:
        """Test handling of errors during camera info retrieval."""
        # モックの準備
        mock_socketio = mock_socketio_class.return_value

        # カメラ情報取得でエラーを発生させる
        self.mock_camera.get_camera_info.side_effect = CameraError("Failed to get camera info")

        # CameraStreamerを初期化
        streamer = CameraStreamer(camera=self.mock_camera)

        # _stream_framesの最初の部分だけを抽出して実行
        with patch.object(streamer, "_stop_event") as mock_stop_event:
            # すぐに停止するようにする
            mock_stop_event.is_set.side_effect = [False, True]

            # カメラが実行中に設定
            self.mock_camera.is_running = True

            # ログメッセージをキャプチャ
            with patch("moro.modules.camera.streaming.logger.error") as mock_logger_error:
                # ストリーミングスレッド関数を実行
                streamer._stream_frames()

                # エラーログが記録されたか確認
                mock_logger_error.assert_called_with(
                    "Failed to get camera info: Failed to get camera info"
                )

        # カメラ情報が送信されていないことを確認
        for call in mock_socketio.emit.call_args_list:
            args = call[0]
            assert args[0] != "camera_info", "カメラ情報がエラーにも関わらず送信されています"
