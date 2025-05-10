"""Camera streaming implementation.

This module provides the main camera streaming functionality using Flask and Flask-SocketIO.
"""

import logging
import threading
import time
from typing import Any, Optional

from flask import Blueprint, Flask, render_template
from flask_socketio import SocketIO

from moro.modules.camera.camera import Camera, CameraError
from moro.modules.camera.emitters import SocketIOFrameEmitter
from moro.modules.camera.processors import DefaultFrameProcessor
from moro.modules.camera.protocols import FrameEmitter, FrameProcessor

# ロガーの設定
logger = logging.getLogger(__name__)
werkzeug_logger = logging.getLogger("werkzeug")
# 明示的にwerkzeugのログレベルをルートロガーから継承
root_logger = logging.getLogger()
werkzeug_logger.setLevel(root_logger.level)


class CameraStreamer:
    """Camera streaming class.

    This class handles streaming camera frames to web clients using Flask and Flask-SocketIO.

    Attributes:
        camera: The Camera instance used for streaming.
        fps: Target frames per second for the stream.
        host: Host address to bind the server to.
        port: Port to run the server on.
        frame_processor: Component that processes video frames.
        frame_emitter: Component that emits frames to clients.
        app: Flask application instance.
        socketio: SocketIO server instance.
        stream_active: Boolean indicating if streaming is active.
        preload_frames: Whether to preload and cache frames before client connections.
    """

    def __init__(
        self,
        camera: Camera,
        fps: int = 15,
        quality: int = 70,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 5000,
        resolution: Optional[tuple[int, int]] = None,
        grayscale: bool = False,
        motion_detection: bool = False,
        motion_threshold: float = 0.005,
        motion_max_skip_frames: int = 30,
        preload_frames: bool = True,
        frame_processor: Optional[FrameProcessor] = None,
        frame_emitter: Optional[FrameEmitter] = None,
    ) -> None:
        """Initialize a CameraStreamer instance.

        Args:
            camera: Camera instance to stream from.
            fps: Target frames per second. Defaults to 15.
            quality: JPEG compression quality (0-100). Defaults to 70.
            host: Host address to bind the server to. Defaults to "0.0.0.0".
            port: Port to run the server on. Defaults to 5000.
            resolution: Optional tuple (width, height) to resize frames. Defaults to None.
            grayscale: Whether to convert frames to grayscale. Defaults to False.
            motion_detection: Whether to enable motion-based frame skipping. Defaults to False.
            motion_threshold: Threshold for motion detection (0.0-1.0). Defaults to 0.005.
            motion_max_skip_frames: Maximum number of consecutive frames to skip. Defaults to 30.
            preload_frames: Whether to preload frames before client connections. Defaults to True.
            frame_processor: Custom frame processor. If None, a DefaultFrameProcessor is used.
            frame_emitter: Custom frame emitter. If None, a SocketIOFrameEmitter is used.
        """
        self.camera = camera
        self.fps = fps
        self.quality = quality
        self.host = host
        self.port = port
        self.resolution = resolution
        self.grayscale = grayscale
        self.motion_detection = motion_detection
        self.motion_threshold = motion_threshold
        self.motion_max_skip_frames = motion_max_skip_frames
        self.preload_frames = preload_frames

        # Flask setup
        self.app = Flask(__name__)

        # カメラモジュール用のBlueprintを作成
        self.camera_bp = Blueprint(
            "camera",
            __name__,
            template_folder="templates",
            static_folder="static",
            static_url_path="/camera/static",
        )

        # ルートの設定 - Blueprintを登録する前に行う
        self._setup_routes()

        # Blueprintをアプリケーションに登録
        self.app.register_blueprint(self.camera_bp, url_prefix="")

        # SocketIO setupを修正
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode="threading",
            logger=False,  # SocketIOのロガーを無効化
            engineio_logger=False,  # EngineIOのロガーを無効化
            ping_timeout=60,
        )

        # Initialize frame processor and emitter
        self.frame_processor = frame_processor or DefaultFrameProcessor(
            resolution=resolution,
            grayscale=grayscale,
            motion_detection=motion_detection,
            motion_threshold=motion_threshold,
            motion_max_skip_frames=motion_max_skip_frames,
        )

        self.frame_emitter = frame_emitter or SocketIOFrameEmitter(
            socketio=self.socketio,
            quality=quality,
        )

        # Streaming state
        self.stream_active = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # SocketIOイベントハンドラの設定
        self._setup_socketio_handlers()

        logger.debug(
            f"Initialized CameraStreamer with fps={fps}, "
            f"host={host}, port={port}, "
            f"frame_processor={self.frame_processor.__class__.__name__}, "
            f"frame_emitter={self.frame_emitter.__class__.__name__}, "
            f"preload_frames={preload_frames}"
        )

    def _setup_routes(self) -> None:
        """Set up Flask routes for the web interface."""

        @self.camera_bp.route("/")
        def index() -> str:  # type: ignore [unused-ignore]
            """Serve the index page with video streaming UI."""
            return render_template("stream.html")

    def _setup_socketio_handlers(self) -> None:
        """Set up SocketIO event handlers for client connections.

        This method establishes handlers for client connection events,
        ensuring that newly connected clients immediately receive
        the latest camera frame and camera information.
        """

        @self.socketio.on("connect")
        def handle_connect() -> None:  # type: ignore [unused-ignore]
            """Handle new client connection.

            When a client connects, immediately send the cached frame
            and camera information to provide instant feedback.
            """
            logger.info("New client connected")
            try:
                # Send cached frame if available
                if hasattr(self.frame_emitter, "cached_frame") and self.frame_emitter.cached_frame:
                    logger.debug("Sending cached frame to new client")
                    self.socketio.emit("frame", {"frame": self.frame_emitter.cached_frame})

                # Send camera info if available
                if hasattr(self.frame_emitter, "camera_info") and self.frame_emitter.camera_info:
                    logger.debug("Sending camera info to new client")
                    self.socketio.emit("camera_info", self.frame_emitter.camera_info)
            except Exception as e:
                logger.error(f"Error handling client connection: {e!s}")

    def _stream_frames(self) -> None:
        """Stream frames from the camera at the specified FPS.

        This method runs in a separate thread and emits camera frames to connected clients.
        """
        if not self.camera.is_running:
            logger.info("Starting camera before streaming")
            self.camera.start()

        logger.info("Frame streaming started")

        # Send initial camera info
        try:
            camera_info = self.camera.get_camera_info()
            # Add streaming configuration to camera info
            camera_info.update(self._get_streaming_config())
            # Emit camera info to clients
            self.frame_emitter.emit_camera_info(camera_info)
        except CameraError as e:
            logger.error(f"Failed to get camera info: {e!s}")

        frame_interval = 1.0 / self.fps
        frame_count = 0
        last_fps_log = time.time()

        while not self._stop_event.is_set():
            start_time = time.time()

            try:
                frame = self.camera.read_frame()
                if frame is not None:
                    # Process frame according to settings
                    processed_frame = self.frame_processor.process_frame(frame)

                    # Skip frame if processor indicated no significant change
                    if processed_frame is None:
                        process_time = time.time() - start_time
                        sleep_time = max(0, frame_interval - process_time)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue

                    # Emit processed frame to clients
                    self.frame_emitter.emit_frame(processed_frame)

                    # Count frames for FPS logging
                    frame_count += 1
                    if time.time() - last_fps_log >= 10.0:  # Log every 10 seconds
                        actual_fps = frame_count / (time.time() - last_fps_log)

                        # Get monitoring info if available
                        skipped_info = ""
                        if hasattr(self.frame_processor, "last_change_ratio"):
                            ratio = self.frame_processor.last_change_ratio
                            skipped_info = f", motion_ratio: {ratio:.6f}"

                            has_skip = hasattr(self.frame_processor, "skipped_frames_count")
                            if has_skip and self.frame_processor.skipped_frames_count > 0:
                                skip_count = self.frame_processor.skipped_frames_count
                                skipped_info += f", skipped: {skip_count}"

                        logger.info(f"Streaming at {actual_fps:.1f} FPS{skipped_info}")
                        frame_count = 0
                        last_fps_log = time.time()

            except CameraError as e:
                logger.error(f"Error reading frame: {e!s}")
                self._stop_event.set()
                break
            except Exception as e:
                logger.error(f"Unexpected error in streaming thread: {e!s}")
                self._stop_event.set()
                break

            # Calculate sleep time to maintain target FPS
            process_time = time.time() - start_time
            sleep_time = max(0, frame_interval - process_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Frame streaming stopped")

    def _get_streaming_config(self) -> dict[str, Any]:
        """Get the current streaming configuration.

        Returns:
            Dictionary with current streaming settings.
        """
        # Get frame processor config
        processor_config: dict[str, Any] = {}

        if isinstance(self.frame_processor, DefaultFrameProcessor):
            processor = self.frame_processor
            processor_config = {
                "stream_resolution": str(processor.resolution)
                if processor.resolution
                else "Original",
                "stream_grayscale": processor.grayscale,
                "stream_motion_detection": processor.motion_detection,
                "stream_motion_threshold": (
                    processor.motion_threshold if processor.motion_detection else "N/A"
                ),
                "stream_motion_max_skip": (
                    processor.motion_max_skip_frames if processor.motion_detection else "N/A"
                ),
            }

        # Get emitter config
        emitter_config: dict[str, Any] = {}
        if isinstance(self.frame_emitter, SocketIOFrameEmitter):
            emitter_config = {
                "stream_quality": self.frame_emitter.quality,
            }

        # Base config
        config = {
            "stream_fps": self.fps,
        }

        # Merge configs
        config.update(processor_config)
        config.update(emitter_config)

        return config

    def start(self) -> None:
        """Start the streaming server and begin streaming frames.

        Raises:
            RuntimeError: If streaming is already active.
        """
        if self.stream_active:
            logger.warning("Streaming is already active")
            return

        logger.info(f"Starting camera streamer on {self.host}:{self.port}")

        # Reset stop event
        self._stop_event.clear()

        # Ensure camera is running
        if not self.camera.is_running:
            logger.info("Starting camera for streaming")
            self.camera.start()

        # Preload initial frame if enabled
        if self.preload_frames:
            try:
                logger.info("Preloading initial frame...")
                # Capture a few frames to let auto-exposure stabilize
                for _ in range(3):
                    self.camera.read_frame()
                    time.sleep(0.1)

                # Capture the initial frame for caching
                initial_frame = self.camera.read_frame()
                if initial_frame is not None:
                    processed_frame = self.frame_processor.process_frame(initial_frame)
                    if processed_frame is not None:
                        # Emit frame to set up cache
                        self.frame_emitter.emit_frame(processed_frame)
                        logger.info("Initial frame preloaded successfully")
                    else:
                        logger.warning("Failed to process initial frame")
                else:
                    logger.warning("Failed to capture initial frame")
            except Exception as e:
                logger.error(f"Error preloading initial frame: {e!s}")

        # Start streaming thread
        self._stream_thread = threading.Thread(target=self._stream_frames, daemon=True)
        self._stream_thread.start()

        self.stream_active = True

        # Start Flask-SocketIO server
        logger.info("Starting Flask-SocketIO server")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=False)

    def stop(self) -> None:
        """Stop streaming and shut down the server."""
        if not self.stream_active:
            logger.warning("Streaming is not active")
            return

        logger.info("Stopping camera streamer")

        # Signal the streaming thread to stop
        self._stop_event.set()

        # Wait for thread to finish if it exists
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5.0)

        # Stop the server
        self.stream_active = False

        logger.info("Camera streamer stopped")
