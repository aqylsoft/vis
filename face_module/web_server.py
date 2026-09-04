"""Web server for streaming face recognition to mobile devices."""

import os
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

from .detector import FaceDetector
from .face_db import FaceDatabase
from .ui import ModernUI

# Try to import anthropic for AI features
try:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    HAS_AI = True
except ImportError:
    HAS_AI = False


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="theme-color" content="#000000">
    <title>Face Module</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        html, body {
            background: #000;
            height: 100%;
            width: 100%;
            overflow: hidden;
        }
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        #stream {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .status-dot {
            position: fixed;
            top: 15px;
            right: 15px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff4444;
            box-shadow: 0 0 10px #ff4444;
        }
        .status-dot.connected {
            background: #44ff44;
            box-shadow: 0 0 10px #44ff44;
        }
        .fullscreen-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            border: none;
            color: #fff;
            font-size: 24px;
            cursor: pointer;
            backdrop-filter: blur(10px);
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .fullscreen-btn:active {
            background: rgba(255,255,255,0.4);
        }
    </style>
</head>
<body>
    <img id="stream" src="/video_feed" alt="Stream">
    <div class="status-dot" id="status"></div>
    <button class="fullscreen-btn" onclick="toggleFullscreen()">⛶</button>

    <script>
        const img = document.getElementById('stream');
        const status = document.getElementById('status');

        img.onload = () => status.classList.add('connected');
        img.onerror = () => {
            status.classList.remove('connected');
            setTimeout(() => img.src = '/video_feed?' + Date.now(), 2000);
        };

        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        }

        // Auto-fullscreen on tap (mobile)
        document.body.addEventListener('click', (e) => {
            if (e.target === img && !document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(() => {});
            }
        });
    </script>
</body>
</html>
"""


class VideoStreamer:
    """Handles video capture and processing in a separate thread."""

    def __init__(
        self,
        camera_id: int = 0,
        mode: str = "watch",
        tolerance: float = 0.6,
        ai_cooldown: float = 30.0,
        language: str = "ru",
    ):
        self.camera_id = camera_id
        self.mode = mode
        self.tolerance = tolerance
        self.ai_cooldown = ai_cooldown
        self.language = language

        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Face recognition setup
        self.db = FaceDatabase()
        self.detector = FaceDetector(self.db, tolerance=tolerance)
        self.ui = ModernUI(title="Face Module")

        # AI setup
        self.client = None
        if mode == "greet" and HAS_AI:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)

        # State
        self.last_greeted: dict[str, float] = {}
        self.moods: dict[str, str] = {}

    def start(self) -> bool:
        """Start the video capture thread."""
        if self.db.is_empty():
            print("Warning: No faces registered")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        """Stop the video capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _analyze_face(self, frame: np.ndarray, name: str) -> Optional[str]:
        """Analyze face with AI and return mood."""
        if not self.client:
            return None

        try:
            from .ai_greeter import analyze_mood_and_greet
            analysis = analyze_mood_and_greet(
                self.client, frame, name, self.language
            )
            print(f"[AI] {name}: {analysis.mood} - {analysis.greeting}")
            self.ui.show_notification(analysis.greeting, "success")
            return analysis.mood
        except Exception as e:
            print(f"[AI Error] {e}")
            return None

    def _capture_loop(self) -> None:
        """Main capture and processing loop."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            self.running = False
            return

        print(f"Camera {self.camera_id} opened")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Mirror
            frame = cv2.flip(frame, 1)

            # Recognize faces
            results = self.detector.recognize_faces(frame)

            now = time.time()

            # Process detections
            for (top, right, bottom, left), name, distance in results:
                if name and self.mode == "greet":
                    # AI greeting mode
                    if name not in self.last_greeted or (now - self.last_greeted[name]) > self.ai_cooldown:
                        mood = self._analyze_face(frame, name)
                        if mood:
                            self.moods[name] = mood
                        self.last_greeted[name] = now
                elif name:
                    # Regular watch mode
                    if name not in self.last_greeted or (now - self.last_greeted[name]) > 5.0:
                        print(f"[DETECTED] {name} (distance: {distance:.3f})")
                        self.ui.show_notification(f"Detected: {name}", "success")
                        self.last_greeted[name] = now

            # Update UI
            self.ui.update_faces(results, self.moods)
            status_right = time.strftime("%H:%M:%S")
            frame = self.ui.render(frame, status_right=status_right)

            # Store frame for streaming
            with self.lock:
                self.frame = frame.copy()

            time.sleep(0.03)  # ~30 FPS

        cap.release()

    def get_frame(self) -> bytes:
        """Get current frame as JPEG bytes."""
        with self.lock:
            if self.frame is None:
                # Generate placeholder frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, "Loading camera...", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2
                )
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            _, jpeg = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes()


def create_app(streamer: VideoStreamer, host: str, port: int) -> Flask:
    """Create Flask application."""
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            title="Face Module",
            mode=streamer.mode,
            host=host,
            port=port,
        )

    @app.route('/snapshot')
    def snapshot():
        """Single frame snapshot for testing."""
        frame = streamer.get_frame()
        return Response(frame, mimetype='image/jpeg')

    @app.route('/video_feed')
    def video_feed():
        def generate() -> Generator[bytes, None, None]:
            while streamer.running:
                frame = streamer.get_frame()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.033)

        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Access-Control-Allow-Origin': '*',
            }
        )

    return app


def run_web_server(
    camera_id: int = 0,
    mode: str = "watch",
    host: str = "0.0.0.0",
    port: int = 5000,
    tolerance: float = 0.6,
    ai_cooldown: float = 30.0,
    language: str = "ru",
) -> int:
    """Run the web server for mobile access.

    Args:
        camera_id: Camera device ID.
        mode: "watch" or "greet" (AI mode).
        host: Host to bind to (0.0.0.0 for all interfaces).
        port: Port number.
        tolerance: Face matching tolerance.
        ai_cooldown: Seconds between AI greetings.
        language: Language for AI greetings.

    Returns:
        Exit code.
    """
    # Get local IP for display
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    streamer = VideoStreamer(
        camera_id=camera_id,
        mode=mode,
        tolerance=tolerance,
        ai_cooldown=ai_cooldown,
        language=language,
    )

    if not streamer.start():
        return 1

    app = create_app(streamer, local_ip, port)

    print("=" * 50)
    print(f"Web server running!")
    print(f"Open in browser: http://{local_ip}:{port}")
    print(f"Mode: {mode}")
    print("=" * 50)

    try:
        app.run(host=host, port=port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        streamer.stop()

    return 0
