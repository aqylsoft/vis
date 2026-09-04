"""AI-powered mood analysis and personalized greetings using Claude Vision."""

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
import cv2
import numpy as np
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from .detector import FaceDetector
from .face_db import FaceDatabase
from .ui import ModernUI


@dataclass
class MoodAnalysis:
    """Result of mood analysis."""
    mood: str
    confidence: str
    greeting: str
    observation: str


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode OpenCV frame to base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.standard_b64encode(buffer).decode("utf-8")


def analyze_mood_and_greet(
    client: anthropic.Anthropic,
    frame: np.ndarray,
    user_name: str,
    language: str = "ru",
) -> MoodAnalysis:
    """Analyze mood from frame and generate personalized greeting.

    Args:
        client: Anthropic client.
        frame: BGR image from OpenCV.
        user_name: Name of the recognized person.
        language: Language for response (ru/en).

    Returns:
        MoodAnalysis with mood, greeting, and observations.
    """
    image_data = encode_frame_to_base64(frame)

    lang_instruction = "Отвечай на русском языке." if language == "ru" else "Respond in English."

    prompt = f"""Ты дружелюбный ИИ-ассистент с камерой. Ты видишь человека по имени {user_name}.

Проанализируй изображение и определи:
1. Настроение/эмоцию человека (счастливый, уставший, сосредоточенный, грустный, нейтральный, и т.д.)
2. Уверенность в оценке (высокая/средняя/низкая)
3. Краткое наблюдение (что заметил - выражение лица, поза, освещение, время суток если видно)

Затем сгенерируй короткое (1-2 предложения) персонализированное приветствие, учитывая:
- Имя человека
- Его текущее настроение
- Может быть совет или комплимент

{lang_instruction}

Ответь СТРОГО в формате:
MOOD: <настроение одним-двумя словами>
CONFIDENCE: <высокая/средняя/низкая>
OBSERVATION: <краткое наблюдение>
GREETING: <персонализированное приветствие>"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    # Parse response
    text = response.content[0].text

    mood = "нейтральный"
    confidence = "средняя"
    observation = ""
    greeting = f"Привет, {user_name}!"

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("MOOD:"):
            mood = line[5:].strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line[11:].strip()
        elif line.startswith("OBSERVATION:"):
            observation = line[12:].strip()
        elif line.startswith("GREETING:"):
            greeting = line[9:].strip()

    return MoodAnalysis(
        mood=mood,
        confidence=confidence,
        greeting=greeting,
        observation=observation,
    )


def run_ai_greeter(
    camera_id: int = 0,
    greet_cooldown: float = 30.0,
    tolerance: float = 0.6,
    language: str = "ru",
    show_analysis: bool = True,
) -> int:
    """Run AI greeter that analyzes mood and greets recognized users.

    Args:
        camera_id: Camera device ID.
        greet_cooldown: Seconds between greetings for same person.
        tolerance: Face matching tolerance.
        language: Language for greetings (ru/en).
        show_analysis: Print detailed analysis to console.

    Returns:
        Exit code.
    """
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Export it: export ANTHROPIC_API_KEY='your-key-here'")
        return 1

    db = FaceDatabase()
    if db.is_empty():
        print("Error: No faces registered. Use 'make face-register' first.")
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    detector = FaceDetector(db, tolerance=tolerance)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return 1

    print("AI Greeter active!")
    print("Press F for fullscreen, Q to quit")
    print("-" * 50)

    # Create window without toolbar
    cv2.namedWindow("AI Greeter", cv2.WINDOW_GUI_NORMAL)

    ui = ModernUI(title="AI Greeter")
    last_greeted: dict[str, float] = {}
    moods: dict[str, str] = {}
    current_greeting: Optional[str] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror
            frame = cv2.flip(frame, 1)

            # Recognize faces
            results = detector.recognize_faces(frame)

            now = time.time()

            for (top, right, bottom, left), name, distance in results:
                if name:
                    # Check if should greet
                    if name not in last_greeted or (now - last_greeted[name]) > greet_cooldown:
                        print(f"\n[{time.strftime('%H:%M:%S')}] Analyzing {name}...")

                        try:
                            analysis = analyze_mood_and_greet(
                                client, frame, name, language
                            )
                            moods[name] = analysis.mood
                            last_greeted[name] = now
                            current_greeting = analysis.greeting

                            # Show notification
                            ui.show_notification(analysis.greeting, "success")

                            # Print to console
                            print(f"  Mood: {analysis.mood} ({analysis.confidence})")
                            if show_analysis and analysis.observation:
                                print(f"  Observation: {analysis.observation}")
                            print(f"  >>> {analysis.greeting}")

                        except Exception as e:
                            print(f"  Error: {e}")
                            ui.show_notification(f"Error: {e}", "error")

            # Update UI state
            ui.update_faces(results, moods)

            # Render modern UI
            status_right = time.strftime("%H:%M:%S")
            frame = ui.render(frame, status_left="", status_right=status_right)

            cv2.imshow("AI Greeter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                cv2.setWindowProperty("AI Greeter", cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty("AI Greeter", cv2.WND_PROP_FULLSCREEN) == 0 else cv2.WINDOW_NORMAL)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0
