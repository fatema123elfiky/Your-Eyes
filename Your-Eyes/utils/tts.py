import pyttsx3
from collections import Counter
from typing import List, Dict
import threading


class TextToSpeech:

    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.engine = None
        self.rate = rate
        self.volume = volume
        self.initialize_engine()

    def initialize_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)

            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break

            print("✅ TTS engine initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")
            self.engine = None

    def speak(self, text: str, blocking: bool = True):
        if not self.engine:
            print(f"TTS not available. Would say: {text}")
            return

        try:
            if blocking:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                thread = threading.Thread(target=self._speak_async, args=(text,))
                thread.daemon = True
                thread.start()
        except Exception as e:
            print(f"Error speaking: {e}")

    def _speak_async(self, text: str):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in async speech: {e}")

    def set_rate(self, rate: int):
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)

    def set_volume(self, volume: float):
        self.volume = volume
        if self.engine:
            self.engine.setProperty('volume', volume)

    def generate_simple_description(self, detected_objects: List[Dict]) -> str:
        if not detected_objects:
            return "No objects detected in the scene."

        labels = [obj["label"] for obj in detected_objects]
        counts = Counter(labels)

        parts = []
        for label, count in counts.items():
            if count == 1:
                parts.append(f"one {label}")
            else:
                parts.append(f"{count} {label}s")

        description = "I see " + ", ".join(parts) + "."
        return description

    def generate_detailed_description(
        self,
        detected_objects: List[Dict],
        image_width: int,
        image_height: int,
        include_distance: bool = True,
        include_position: bool = True
    ) -> str:
        if not detected_objects:
            return "No objects detected in the scene."

        priority_objects = [obj for obj in detected_objects if obj.get("is_priority", False)]
        other_objects = [obj for obj in detected_objects if not obj.get("is_priority", False)]

        description_parts = []

        if priority_objects:
            description_parts.append("Warning!")
            for obj in priority_objects[:3]:
                obj_desc = self._describe_object(
                    obj, image_width, image_height,
                    include_distance, include_position
                )
                description_parts.append(obj_desc)

        if other_objects:
            if priority_objects:
                description_parts.append("Also,")

            labels = [obj["label"] for obj in other_objects]
            counts = Counter(labels)

            parts = []
            for label, count in list(counts.items())[:5]:
                if count == 1:
                    parts.append(f"one {label}")
                else:
                    parts.append(f"{count} {label}s")

            if parts:
                description_parts.append("I see " + ", ".join(parts))

        return ". ".join(description_parts) + "."

    def _describe_object(
        self,
        obj: Dict,
        image_width: int,
        image_height: int,
        include_distance: bool,
        include_position: bool
    ) -> str:
        label = obj["label"]
        parts = [f"A {label}"]

        if include_distance:
            distance = self._estimate_distance(obj["area"], image_width, image_height)
            parts.append(distance)

        if include_position:
            position = self._get_position(obj["center_x"], image_width)
            parts.append(position)

        return " ".join(parts)

    def _estimate_distance(self, area: float, image_width: int, image_height: int) -> str:
        image_area = image_width * image_height
        relative_size = area / image_area

        if relative_size > 0.3:
            return "very close"
        elif relative_size > 0.15:
            return "close"
        elif relative_size > 0.05:
            return "at medium distance"
        else:
            return "far away"

    def _get_position(self, center_x: float, image_width: int) -> str:
        relative_x = center_x / image_width

        if relative_x < 0.33:
            return "on your left"
        elif relative_x < 0.67:
            return "in front of you"
        else:
            return "on your right"

