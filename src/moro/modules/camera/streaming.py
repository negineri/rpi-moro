"""Streaming module for USB camera.

This module provides functionality for streaming camera frames over HTTP
using Flask and Flask-SocketIO.
"""

import base64
import logging
import threading
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from flask import Flask
from flask_socketio import SocketIO

from moro.modules.camera.camera import Camera, CameraError

logger = logging.getLogger(__name__)


class CameraStreamer:
    """Camera streaming class.

    This class handles streaming camera frames to web clients using Flask and Flask-SocketIO.

    Attributes:
        camera: The Camera instance used for streaming.
        fps: Target frames per second for the stream.
        quality: JPEG compression quality (0-100).
        resolution: Optional resolution for resizing frames (width, height).
        grayscale: Whether to convert frames to grayscale.
        motion_detection: Whether to use motion detection to skip similar frames.
        motion_threshold: Threshold for motion detection (0.0-1.0).
        motion_max_skip_frames: Maximum number of consecutive frames to skip.
        app: Flask application instance.
        socketio: SocketIO server instance.
        stream_active: Boolean indicating if streaming is active.
    """

    def __init__(
        self,
        camera: Camera,
        fps: int = 15,
        quality: int = 70,
        host: str = "0.0.0.0",
        port: int = 5000,
        resolution: Optional[Tuple[int, int]] = None,
        grayscale: bool = True,
        motion_detection: bool = True,
        motion_threshold: float = 0.005,
        motion_max_skip_frames: int = 30,
    ) -> None:
        """Initialize a CameraStreamer instance.

        Args:
            camera: Camera instance to stream from.
            fps: Target frames per second. Defaults to 15.
            quality: JPEG compression quality (0-100). Defaults to 70.
            host: Host address to bind the server to. Defaults to "0.0.0.0".
            port: Port to run the server on. Defaults to 5000.
            resolution: Optional tuple (width, height) to resize frames. Defaults to None (no resize).
            grayscale: Whether to convert frames to grayscale. Defaults to False.
            motion_detection: Whether to enable motion-based frame skipping. Defaults to False.
            motion_threshold: Threshold for motion detection (0.0-1.0). Defaults to 0.005.
            motion_max_skip_frames: Maximum number of consecutive frames to skip. Defaults to 30.
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

        # Flask setup
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")

        # Streaming state
        self.stream_active = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._previous_frame: Optional[np.ndarray[Any, np.dtype[Any]]] = None
        self._skipped_frames_count: int = 0
        self._last_change_ratio: float = 0.0

        # Set up Flask routes
        self._setup_routes()

        logger.debug(
            f"Initialized CameraStreamer with fps={fps}, quality={quality}, "
            f"host={host}, port={port}, resolution={resolution}, grayscale={grayscale}, "
            f"motion_detection={motion_detection}, motion_threshold={motion_threshold}, "
            f"motion_max_skip_frames={motion_max_skip_frames}"
        )

    def _setup_routes(self) -> None:
        """Set up Flask routes for the web interface."""

        @self.app.route("/")
        def index() -> str:  # type: ignore [unused-ignore]
            """Serve the index page with video streaming UI."""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>カメラストリーミング</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
                    h1 { color: #333; }
                    #video-container { margin: 20px auto; max-width: 800px; }
                    #stream { max-width: 100%; border: 1px solid #ddd; }
                    .control-panel { margin: 20px auto; max-width: 600px; }
                    button { 
                        background-color: #4CAF50; 
                        border: none; 
                        color: white;
                        padding: 10px 20px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 4px;
                    }
                    .camera-info {
                        margin: 20px auto;
                        max-width: 600px;
                        text-align: left;
                        background-color: #f9f9f9;
                        padding: 15px;
                        border-radius: 5px;
                    }
                </style>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        // Connect to the SocketIO server
                        const socket = io.connect(window.location.origin);
                        const streamElement = document.getElementById('stream');
                        const fpsCounter = document.getElementById('fps-counter');
                        const cameraInfo = document.getElementById('camera-info');
                        
                        let frameCount = 0;
                        let lastTime = new Date().getTime();
                        
                        // Receive and display frame
                        socket.on('frame', function(data) {
                            if (streamElement && data.frame) {
                                streamElement.src = 'data:image/jpeg;base64,' + data.frame;
                                
                                // Update FPS counter
                                frameCount++;
                                const now = new Date().getTime();
                                if (now - lastTime >= 1000) {
                                    const fps = frameCount / ((now - lastTime) / 1000);
                                    fpsCounter.innerText = fps.toFixed(1);
                                    frameCount = 0;
                                    lastTime = now;
                                }
                            }
                        });
                        
                        // Receive camera info
                        socket.on('camera_info', function(info) {
                            if (info) {
                                let infoHTML = '<h3>カメラ情報</h3>';
                                infoHTML += '<ul>';
                                for (const [key, value] of Object.entries(info)) {
                                    infoHTML += `<li>${key}: ${value}</li>`;
                                }
                                infoHTML += '</ul>';
                                cameraInfo.innerHTML = infoHTML;
                            }
                        });
                    });
                </script>
            </head>
            <body>
                <h1>ラズベリーパイ カメラストリーム</h1>
                <div id="video-container">
                    <img id="stream" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=">
                </div>
                <div class="control-panel">
                    <p>ストリーミング速度: <span id="fps-counter">0.0</span> FPS</p>
                </div>
                <div id="camera-info" class="camera-info">
                    <h3>カメラ情報</h3>
                    <p>接続中...</p>
                </div>
            </body>
            </html>
            """

    def _process_frame(self, frame: np.ndarray[Any, Any]) -> Optional[np.ndarray[Any, Any]]:
        """Process a frame according to the configured options.

        This method applies resolution changes, grayscale conversion, and motion detection.

        Args:
            frame: The source frame to process.

        Returns:
            The processed frame or None if the frame should be skipped.
        """
        processed_frame = frame.copy()

        # Resize the frame first if resolution is specified
        if self.resolution:
            processed_frame = cv2.resize(processed_frame, self.resolution)

        # Apply motion detection if enabled
        if self.motion_detection and self._previous_frame is not None:
            # Force sending a frame if we've skipped too many
            if self._skipped_frames_count >= self.motion_max_skip_frames:
                logger.debug(f"Forced frame send after {self._skipped_frames_count} skipped frames")
                self._skipped_frames_count = 0
            # Check for motion only if frame shapes match
            elif processed_frame.shape == self._previous_frame.shape:
                # Convert to grayscale for motion detection (reduces noise and color changes)
                gray_current = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                gray_previous = cv2.cvtColor(self._previous_frame, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to reduce noise
                gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
                gray_previous = cv2.GaussianBlur(gray_previous, (5, 5), 0)

                # Compute absolute difference between current and previous frame
                diff = cv2.absdiff(gray_current, gray_previous)

                # Apply threshold to highlight significant changes
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

                # Count non-zero pixels (changed pixels)
                non_zero = np.count_nonzero(thresh)
                total_pixels = thresh.shape[0] * thresh.shape[1]
                change_ratio = non_zero / total_pixels

                # Store for logging
                self._last_change_ratio = change_ratio

                # Skip frame if change is below threshold
                if change_ratio < self.motion_threshold:
                    self._skipped_frames_count += 1
                    # Log every 10 frames to avoid excessive logging
                    if self._skipped_frames_count % 10 == 1:
                        logger.debug(
                            f"Motion below threshold: {change_ratio:.6f} < {self.motion_threshold}, "
                            f"skipped {self._skipped_frames_count} frames"
                        )
                    return None
                if self._skipped_frames_count > 0:
                    logger.debug(
                        f"Motion detected: {change_ratio:.6f} > {self.motion_threshold}, "
                        f"sent frame after {self._skipped_frames_count} skipped frames"
                    )
                self._skipped_frames_count = 0

        # Apply grayscale conversion if requested (after motion detection)
        if self.grayscale:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            # Convert back to 3 channels for consistent processing
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        # Store frame for next comparison (always store original/color frame for motion detection)
        self._previous_frame = (
            frame.copy() if self.resolution is None else cv2.resize(frame, self.resolution)
        )

        return processed_frame

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
            camera_info.update(
                {
                    "stream_fps": self.fps,
                    "stream_quality": self.quality,
                    "stream_resolution": str(self.resolution) if self.resolution else "Original",
                    "stream_grayscale": self.grayscale,
                    "stream_motion_detection": self.motion_detection,
                    "stream_motion_threshold": self.motion_threshold
                    if self.motion_detection
                    else "N/A",
                    "stream_motion_max_skip": self.motion_max_skip_frames
                    if self.motion_detection
                    else "N/A",
                }
            )
            self.socketio.emit("camera_info", camera_info)
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
                    # Process frame according to bandwidth reduction settings
                    processed_frame = self._process_frame(frame)

                    # Skip frame if motion detection indicated no significant change
                    if processed_frame is None:
                        process_time = time.time() - start_time
                        sleep_time = max(0, frame_interval - process_time)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue

                    # Encode frame to JPEG
                    _, buffer = cv2.imencode(
                        ".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                    )
                    # Convert to base64 for sending over WebSocket
                    b64_frame = base64.b64encode(buffer).decode("utf-8")
                    self.socketio.emit("frame", {"frame": b64_frame})

                    # Count frames for FPS logging
                    frame_count += 1
                    if time.time() - last_fps_log >= 10.0:  # Log every 10 seconds
                        actual_fps = frame_count / (time.time() - last_fps_log)
                        skipped_info = ""
                        if self.motion_detection:
                            skipped_info = f", motion_ratio: {self._last_change_ratio:.6f}"
                            if self._skipped_frames_count > 0:
                                skipped_info += f", skipped: {self._skipped_frames_count}"
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
