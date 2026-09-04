"""CLI interface for face recognition module."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from .detector import FaceDetector
from .face_db import FaceDatabase


def cmd_register(args: argparse.Namespace) -> int:
    """Register a new face in the database."""
    db = FaceDatabase()
    detector = FaceDetector(db)

    name = args.name
    if not name:
        name = input("Enter name for this face: ").strip()
        if not name:
            print("Error: Name cannot be empty")
            return 1

    if args.image:
        # Register from image file
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return 1

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Cannot read image: {image_path}")
            return 1

        locations = detector.detect_faces(frame)
        if not locations:
            print("Error: No face detected in image")
            return 1
        if len(locations) > 1:
            print(f"Error: Multiple faces ({len(locations)}) detected. Use image with single face.")
            return 1

        encoding = detector.encode_face(frame, locations[0])
        if encoding is None:
            print("Error: Failed to encode face")
            return 1

        idx = db.add_face(name, encoding)
        print(f"Registered face '{name}' (ID: {idx})")

    else:
        # Register from camera
        camera_source = args.camera
        if camera_source.isdigit():
            camera_source = int(camera_source)

        print(f"Registering face for '{name}' from camera...")
        try:
            result = detector.capture_face_from_camera(camera_id=camera_source)
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1

        if result is None:
            print("Cancelled")
            return 1

        frame, encoding = result
        idx = db.add_face(name, encoding)
        print(f"Registered face '{name}' (ID: {idx})")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all registered faces."""
    db = FaceDatabase()

    if db.is_empty():
        print("No faces registered")
        return 0

    names = db.list_faces()
    print(f"Registered faces ({db.count()} total):")
    for name in names:
        count = db.names.count(name)
        suffix = f" ({count} encodings)" if count > 1 else ""
        print(f"  - {name}{suffix}")

    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    """Remove a face from the database."""
    db = FaceDatabase()

    if db.remove_face(args.name):
        print(f"Removed '{args.name}'")
        return 0
    else:
        print(f"Face '{args.name}' not found")
        return 1


def cmd_autolock(args: argparse.Namespace) -> int:
    """Run auto-lock daemon."""
    from .autolock import run_autolock

    return run_autolock(
        camera_id=args.camera,
        lock_timeout=args.timeout,
        show_preview=args.preview,
        tolerance=args.tolerance,
    )


def cmd_greet(args: argparse.Namespace) -> int:
    """Run AI greeter with mood analysis."""
    from .ai_greeter import run_ai_greeter

    return run_ai_greeter(
        camera_id=args.camera,
        greet_cooldown=args.cooldown,
        tolerance=args.tolerance,
        language=args.lang,
        show_analysis=not args.quiet,
    )


def cmd_web(args: argparse.Namespace) -> int:
    """Run web server for mobile access."""
    from .web_server import run_web_server

    return run_web_server(
        camera_id=args.camera,
        mode=args.mode,
        host=args.host,
        port=args.port,
        tolerance=args.tolerance,
        ai_cooldown=args.cooldown,
        language=args.lang,
    )


def cmd_watch(args: argparse.Namespace) -> int:
    """Watch camera for known faces."""
    from .ui import ModernUI

    db = FaceDatabase()

    if db.is_empty():
        print("Error: No faces registered. Use 'register' command first.")
        return 1

    detector = FaceDetector(db, tolerance=args.tolerance)

    # Support both camera ID and URL
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return 1

    # Check if phone camera (URL) - no mirror needed
    is_phone = isinstance(args.camera, str) and args.camera.startswith("http")

    print(f"Watching camera: {args.camera}")
    print("Press F for fullscreen, Q to quit")

    # Create window without toolbar
    cv2.namedWindow("Face Watch", cv2.WINDOW_GUI_NORMAL)

    ui = ModernUI(title="Face Watch")
    last_announced: dict[str, float] = {}
    announce_cooldown = 5.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame (selfie mode) - skip for phone camera
            if not is_phone:
                frame = cv2.flip(frame, 1)

            # Recognize faces
            results = detector.recognize_faces(frame)

            # Announce new detections
            now = time.time()
            for (top, right, bottom, left), name, distance in results:
                if name:
                    if name not in last_announced or (now - last_announced[name]) > announce_cooldown:
                        print(f"[DETECTED] {name} (distance: {distance:.3f})")
                        ui.show_notification(f"Detected: {name}", "success")
                        last_announced[name] = now

            # Update and render UI
            ui.update_faces(results)
            status_right = time.strftime("%H:%M:%S")
            frame = ui.render(frame, status_right=status_right)

            cv2.imshow("Face Watch", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                # Toggle fullscreen
                cv2.setWindowProperty("Face Watch", cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty("Face Watch", cv2.WND_PROP_FULLSCREEN) == 0 else cv2.WINDOW_NORMAL)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="face_module",
        description="Face recognition module - register and detect known faces"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register command
    reg_parser = subparsers.add_parser("register", help="Register a new face")
    reg_parser.add_argument("--name", "-n", help="Name for the face")
    reg_parser.add_argument("--image", "-i", help="Path to image file (optional, uses camera if not provided)")
    reg_parser.add_argument("--camera", "-c", default="0", help="Camera ID (0,1..) or phone URL")

    # List command
    subparsers.add_parser("list", help="List registered faces")

    # Remove command
    rem_parser = subparsers.add_parser("remove", help="Remove a face from database")
    rem_parser.add_argument("name", help="Name to remove")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch camera for known faces")
    watch_parser.add_argument("--camera", "-c", default="0",
                             help="Camera ID (0,1..) or phone URL (http://192.168.x.x:8080/video)")
    watch_parser.add_argument("--tolerance", "-t", type=float, default=0.6,
                             help="Face matching tolerance (default: 0.6, lower = stricter)")

    # Autolock command
    lock_parser = subparsers.add_parser("autolock", help="Auto-lock screen when you leave")
    lock_parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID (default: 0)")
    lock_parser.add_argument("--timeout", "-t", type=float, default=10.0,
                            help="Seconds without face before locking (default: 10)")
    lock_parser.add_argument("--tolerance", type=float, default=0.6,
                            help="Face matching tolerance (default: 0.6)")
    lock_parser.add_argument("--preview", "-p", action="store_true",
                            help="Show camera preview window")

    # AI Greeter command
    greet_parser = subparsers.add_parser("greet", help="AI mood analysis and personalized greetings")
    greet_parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID (default: 0)")
    greet_parser.add_argument("--cooldown", type=float, default=30.0,
                             help="Seconds between greetings for same person (default: 30)")
    greet_parser.add_argument("--tolerance", type=float, default=0.6,
                             help="Face matching tolerance (default: 0.6)")
    greet_parser.add_argument("--lang", "-l", choices=["ru", "en"], default="ru",
                             help="Language for greetings (default: ru)")
    greet_parser.add_argument("--quiet", "-q", action="store_true",
                             help="Don't show detailed observations")

    # Web server command
    web_parser = subparsers.add_parser("web", help="Web server for mobile access")
    web_parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID (default: 0)")
    web_parser.add_argument("--mode", "-m", choices=["watch", "greet"], default="watch",
                           help="Mode: watch (detection only) or greet (AI analysis)")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    web_parser.add_argument("--port", "-p", type=int, default=5000, help="Port (default: 5000)")
    web_parser.add_argument("--tolerance", type=float, default=0.6,
                           help="Face matching tolerance (default: 0.6)")
    web_parser.add_argument("--cooldown", type=float, default=30.0,
                           help="AI greeting cooldown in seconds (default: 30)")
    web_parser.add_argument("--lang", "-l", choices=["ru", "en"], default="ru",
                           help="Language for AI greetings (default: ru)")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "register": cmd_register,
        "list": cmd_list,
        "remove": cmd_remove,
        "watch": cmd_watch,
        "autolock": cmd_autolock,
        "greet": cmd_greet,
        "web": cmd_web,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())