"""Test suite for the Camera module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moro.modules.camera.camera import Camera, CameraError


class TestCamera(unittest.TestCase):
    """Test case for the Camera class."""

    def setUp(self) -> None:
        """Set up test fixtures, if any."""
        self.test_device_id = 0
        self.test_width = 640
        self.test_height = 480
        self.test_fps = 30

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_init(self, mock_video_capture: MagicMock) -> None:
        """Test Camera initialization."""
        camera = Camera(
            device_id=self.test_device_id,
            width=self.test_width,
            height=self.test_height,
            fps=self.test_fps,
        )

        # 初期化時にカメラは開始されていない
        assert camera.device_id == self.test_device_id
        assert camera.width == self.test_width
        assert camera.height == self.test_height
        assert camera.fps == self.test_fps
        assert camera.is_running is False
        assert camera._camera is None  # type: ignore

        # モックが呼び出されていないことを確認
        mock_video_capture.assert_not_called()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_start_already_running(self, mock_video_capture: MagicMock) -> None:
        """Test starting the camera when it's already running."""
        camera = Camera(self.test_device_id)
        camera.is_running = True

        # 既に実行中の場合、何も起こらない
        camera.start()
        mock_video_capture.assert_not_called()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_start_failure(self, mock_video_capture: MagicMock) -> None:
        """Test camera start failure."""
        mock_camera = mock_video_capture.return_value
        mock_camera.isOpened.return_value = False

        camera = Camera(self.test_device_id)

        with pytest.raises(CameraError) as exc_info:
            camera.start()

        assert "Failed to open camera" in str(exc_info.value)
        assert camera.is_running is False
        assert camera._camera is None  # type: ignore
        mock_camera.release.assert_called_once()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_start_exception(self, mock_video_capture: MagicMock) -> None:
        """Test exception during camera start."""
        mock_video_capture.side_effect = Exception("Test exception")

        camera = Camera(self.test_device_id)

        with pytest.raises(CameraError) as exc_info:
            camera.start()

        assert "Error starting camera" in str(exc_info.value)
        assert "Test exception" in str(exc_info.value)
        assert camera.is_running is False
        assert camera._camera is None  # type: ignore

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_stop(self, mock_video_capture: MagicMock) -> None:
        """Test stopping the camera."""
        mock_camera = mock_video_capture.return_value
        mock_camera.isOpened.return_value = True

        camera = Camera(self.test_device_id)
        camera.start()
        assert camera.is_running is True

        camera.stop()
        assert camera.is_running is False
        assert camera._camera is None  # type: ignore
        mock_camera.release.assert_called_once()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_stop_not_running(self, mock_video_capture: MagicMock) -> None:
        """Test stopping the camera when it's not running."""
        camera = Camera(self.test_device_id)
        assert camera.is_running is False

        camera.stop()
        assert camera.is_running is False
        mock_video_capture.assert_not_called()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_read_frame_success(self, mock_video_capture: MagicMock) -> None:
        """Test reading a frame successfully."""
        mock_camera = mock_video_capture.return_value
        mock_camera.isOpened.return_value = True

        # ダミーのフレームを作成
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.read.return_value = (True, dummy_frame)

        camera = Camera(self.test_device_id)
        camera.start()

        frame = camera.read_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        mock_camera.read.assert_called_once()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_read_frame_failure(self, mock_video_capture: MagicMock) -> None:
        """Test failure when reading a frame."""
        mock_camera = mock_video_capture.return_value
        mock_camera.isOpened.return_value = True
        mock_camera.read.return_value = (False, None)

        camera = Camera(self.test_device_id)
        camera.start()

        frame = camera.read_frame()
        assert frame is None
        mock_camera.read.assert_called_once()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_read_frame_not_running(self, mock_video_capture: MagicMock) -> None:
        """Test reading a frame when camera is not running."""
        camera = Camera(self.test_device_id)

        with pytest.raises(CameraError) as exc_info:
            camera.read_frame()

        assert "Camera is not running" in str(exc_info.value)
        mock_video_capture.assert_not_called()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_get_camera_info_not_running(self, mock_video_capture: MagicMock) -> None:
        """Test getting camera info when camera is not running."""
        camera = Camera(self.test_device_id)

        with pytest.raises(CameraError) as exc_info:
            camera.get_camera_info()

        assert "Camera is not running" in str(exc_info.value)
        mock_video_capture.assert_not_called()

    @patch("moro.modules.camera.camera.cv2.VideoCapture")
    def test_camera_context_manager(self, mock_video_capture: MagicMock) -> None:
        """Test camera as a context manager."""
        mock_camera = mock_video_capture.return_value
        mock_camera.isOpened.return_value = True

        with Camera(self.test_device_id) as camera:
            assert camera.is_running is True
            assert camera._camera is not None  # type: ignore

        # コンテキスト終了後、カメラが停止していることを確認
        assert camera.is_running is False
        assert camera._camera is None  # type: ignore
        mock_camera.release.assert_called_once()
