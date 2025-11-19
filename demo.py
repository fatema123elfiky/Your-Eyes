"""
Demo script for Your Eyes - Quick command-line testing
Use this to test the system without the GUI
"""

import argparse
import cv2
from pathlib import Path
from models.youreyes_yolo import YourEyesDetector
from utils.tts import TextToSpeech


def demo_image(image_path: str, model_path: str = "yolov8n.pt", conf: float = 0.5):
    """
    Demo on a single image
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model
        conf: Confidence threshold
    """
    print("=" * 70)
    print("Your Eyes - Image Demo")
    print("=" * 70)
    print()
    
    # Load detector and TTS
    print("Loading model...")
    detector = YourEyesDetector(model_path=model_path, conf_threshold=conf)
    tts = TextToSpeech()
    
    # Load image
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    
    # Detect objects
    detected_objects, annotated_image = detector.process_image(image, conf_threshold=conf)
    
    # Display results
    print()
    print(f"✅ Found {len(detected_objects)} object(s)")
    print()
    
    if detected_objects:
        print("Detected objects:")
        for i, obj in enumerate(detected_objects, 1):
            priority = "⚠️ PRIORITY" if obj.get("is_priority", False) else ""
            print(f"  {i}. {obj['label']:20s} - {obj['confidence']:.2%} {priority}")
        
        # Generate description
        height, width = image.shape[:2]
        description = tts.generate_detailed_description(
            detected_objects,
            image_width=width,
            image_height=height
        )
        
        print()
        print("Audio description:")
        print(f"  \"{description}\"")
        print()
        
        # Speak
        response = input("Speak description? (y/n): ")
        if response.lower() == 'y':
            tts.speak(description)
        
        # Show image
        response = input("Show annotated image? (y/n): ")
        if response.lower() == 'y':
            cv2.imshow("Your Eyes - Detection Results", annotated_image)
            print("Press any key in the image window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save result
        response = input("Save annotated image? (y/n): ")
        if response.lower() == 'y':
            output_path = Path(image_path).stem + "_detected.jpg"
            cv2.imwrite(output_path, annotated_image)
            print(f"✅ Saved to: {output_path}")
    
    else:
        print("No objects detected. Try lowering the confidence threshold.")
    
    print()


def demo_webcam(model_path: str = "yolov8n.pt", conf: float = 0.5, camera: int = 0):
    """
    Demo with webcam
    
    Args:
        model_path: Path to YOLO model
        conf: Confidence threshold
        camera: Camera index
    """
    print("=" * 70)
    print("Your Eyes - Webcam Demo")
    print("=" * 70)
    print()
    
    # Load detector and TTS
    print("Loading model...")
    detector = YourEyesDetector(model_path=model_path, conf_threshold=conf)
    tts = TextToSpeech()
    
    # Open webcam
    print(f"Opening camera {camera}...")
    cap = cv2.VideoCapture(camera)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera}")
        return
    
    print()
    print("✅ Webcam opened successfully")
    print()
    print("Controls:")
    print("  Q - Quit")
    print("  S - Speak current description")
    print("  SPACE - Pause/Resume")
    print()
    
    paused = False
    last_description = ""
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Detect objects
            detected_objects, annotated_frame = detector.process_image(
                frame,
                conf_threshold=conf
            )
            
            # Generate description
            if detected_objects:
                height, width = frame.shape[:2]
                last_description = tts.generate_simple_description(detected_objects)
            else:
                last_description = "No objects detected"
            
            # Add text overlay
            cv2.putText(
                annotated_frame,
                f"Objects: {len(detected_objects)} | Q=Quit S=Speak SPACE=Pause",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                annotated_frame,
                last_description[:80],  # Truncate if too long
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Show frame
            cv2.imshow("Your Eyes - Webcam Demo", annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            print(f"Speaking: {last_description}")
            tts.speak(last_description, blocking=False)
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Demo completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your Eyes Demo")
    
    parser.add_argument(
        "mode",
        choices=["image", "webcam"],
        help="Demo mode: image or webcam"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file (for image mode)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model weights"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (for webcam mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "image":
        if not args.image:
            print("❌ Error: --image required for image mode")
            print("Example: python demo.py image --image path/to/image.jpg")
        else:
            demo_image(args.image, args.model, args.conf)
    
    elif args.mode == "webcam":
        demo_webcam(args.model, args.conf, args.camera)

