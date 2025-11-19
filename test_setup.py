"""
Test script to verify Your Eyes setup
Run this to check if all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("=" * 60)
    print("Testing Your Eyes Setup")
    print("=" * 60)
    print()
    
    packages = {
        'ultralytics': 'YOLO object detection',
        'streamlit': 'Web GUI framework',
        'cv2': 'OpenCV for image processing',
        'pyttsx3': 'Text-to-speech',
        'torch': 'PyTorch deep learning',
        'numpy': 'Numerical computing',
        'PIL': 'Image processing'
    }
    
    failed = []
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✅ {package:15s} - {description}")
        except ImportError as e:
            print(f"❌ {package:15s} - FAILED: {e}")
            failed.append(package)
    
    print()
    
    if failed:
        print(f"❌ {len(failed)} package(s) failed to import")
        print(f"   Failed: {', '.join(failed)}")
        print()
        print("To fix, run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All imports successful!")
        return True


def test_yolo():
    """Test YOLO model loading"""
    print("\n" + "=" * 60)
    print("Testing YOLO Model")
    print("=" * 60)
    print()
    
    try:
        from ultralytics import YOLO
        print("Loading YOLOv8 nano model...")
        model = YOLO("yolov8n.pt")
        print(f"✅ YOLO model loaded successfully")
        print(f"   Model: yolov8n.pt")
        print(f"   Classes: {len(model.names)}")
        return True
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        print("   The model will be auto-downloaded on first run")
        return False


def test_tts():
    """Test text-to-speech"""
    print("\n" + "=" * 60)
    print("Testing Text-to-Speech")
    print("=" * 60)
    print()
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("✅ TTS engine initialized successfully")
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"   Available voices: {len(voices)}")
        
        return True
    except Exception as e:
        print(f"❌ TTS initialization failed: {e}")
        return False


def test_opencv():
    """Test OpenCV"""
    print("\n" + "=" * 60)
    print("Testing OpenCV")
    print("=" * 60)
    print()
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test webcam availability (optional)
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✅ Webcam detected and accessible")
                cap.release()
            else:
                print("⚠️  No webcam detected (optional for image mode)")
        except:
            print("⚠️  Could not test webcam (optional)")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False


def test_pytorch():
    """Test PyTorch and CUDA availability"""
    print("\n" + "=" * 60)
    print("Testing PyTorch")
    print("=" * 60)
    print()
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU will be used for faster inference")
        else:
            print("ℹ️  CUDA not available - will use CPU")
            print("   (This is fine, just slower)")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False


def main():
    """Run all tests"""
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("YOLO", test_yolo()))
    results.append(("TTS", test_tts()))
    results.append(("OpenCV", test_opencv()))
    results.append(("PyTorch", test_pytorch()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print()
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:15s}: {status}")
    
    print()
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    if passed_count == total_count:
        print("🎉 All tests passed! You're ready to run the app.")
        print()
        print("Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Open your browser to http://localhost:8501")
        print("3. Go to Settings and click 'Load/Reload Model'")
        print("4. Try Image Mode with a test image")
        print()
        return 0
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed")
        print()
        print("Please fix the issues above before running the app.")
        print("Run: pip install -r requirements.txt")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

