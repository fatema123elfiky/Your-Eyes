
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path

from models.youreyes_yolo import YourEyesDetector
from utils.tts import TextToSpeech


st.set_page_config(
    page_title="Your Eyes - Assistive Vision",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


def apply_theme():
    if st.session_state.dark_mode:
        theme_css = """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #1a1a1a;
                color: #e0e0e0;
            }

            [data-testid="stSidebar"] {
                background-color: #2d2d2d;
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: #e0e0e0;
            }

            .main-header {
                font-size: 3rem;
                font-weight: bold;
                color: #F48FB1 !important;
                text-align: center;
                margin-bottom: 1rem;
            }

            .sub-header {
                font-size: 1.5rem;
                color: #B0BEC5 !important;
                text-align: center;
                margin-bottom: 2rem;
            }

            .priority-warning {
                background-color: #4a2c2c !important;
                border-left: 5px solid #EF5350 !important;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 5px;
            }

            .priority-warning p, .priority-warning span, .priority-warning div,
            .priority-warning strong, .priority-warning em, .priority-warning li {
                color: #ffcdd2 !important;
            }

            .detection-box {
                background-color: #4a2d3e !important;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
            }

            .detection-box p, .detection-box span, .detection-box div,
            .detection-box strong, .detection-box em, .detection-box li {
                color: #F8BBD0 !important;
            }

            .stButton>button {
                font-size: 1.2rem;
                padding: 0.75rem 2rem;
                border-radius: 10px;
                background-color: #C2185B !important;
                color: white !important;
                border: none;
            }

            .stButton>button:hover {
                background-color: #AD1457 !important;
            }

            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
            .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #F48FB1 !important;
            }

            .stMarkdown p, .stMarkdown li, .stMarkdown span,
            .stMarkdown div, .stMarkdown strong, .stMarkdown em {
                color: #e0e0e0 !important;
            }

            /* Input fields */
            .stTextInput input, .stNumberInput input, .stSelectbox select {
                background-color: #3d3d3d !important;
                color: #e0e0e0 !important;
                border-color: #555 !important;
            }

            /* Radio buttons and checkboxes */
            .stRadio label, .stCheckbox label {
                color: #e0e0e0 !important;
            }

            /* Slider */
            .stSlider label {
                color: #e0e0e0 !important;
            }

            /* Tab content */
            [data-testid="stTab"] p, [data-testid="stTab"] span,
            [data-testid="stTab"] div, [data-testid="stTab"] li,
            [data-testid="stTab"] strong, [data-testid="stTab"] em {
                color: #e0e0e0 !important;
            }

            /* Alert/Info boxes */
            .stAlert p, .stAlert span, .stAlert div, .stAlert li {
                color: #e0e0e0 !important;
            }
        </style>
        """
    else:
        theme_css = """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff;
                color: #000000;
            }

            [data-testid="stSidebar"] {
                background-color: #f0f2f6;
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: #000000;
            }

            .main-header {
                font-size: 3rem;
                font-weight: bold;
                color: #C2185B !important;
                text-align: center;
                margin-bottom: 1rem;
            }

            .sub-header {
                font-size: 1.5rem;
                color: #000000 !important;
                text-align: center;
                margin-bottom: 2rem;
            }

            .priority-warning {
                background-color: #FFEBEE !important;
                border-left: 5px solid #F44336 !important;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 5px;
            }

            .priority-warning p, .priority-warning span, .priority-warning div,
            .priority-warning strong, .priority-warning em, .priority-warning li {
                color: #000000 !important;
            }

            .detection-box {
                background-color: #FCE4EC !important;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
            }

            .detection-box p, .detection-box span, .detection-box div,
            .detection-box strong, .detection-box em, .detection-box li {
                color: #000000 !important;
            }

            .stButton>button {
                font-size: 1.2rem;
                padding: 0.75rem 2rem;
                border-radius: 10px;
                background-color: #C2185B !important;
                color: white !important;
            }

            .stButton>button:hover {
                background-color: #AD1457 !important;
            }

            /* Text colors for light mode - BLACK */
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
            .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #C2185B !important;
            }

            .stMarkdown p, .stMarkdown li, .stMarkdown span,
            .stMarkdown div, .stMarkdown strong, .stMarkdown em {
                color: #000000 !important;
            }

            /* Sidebar text black */
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
                color: #000000 !important;
            }

            /* Input labels and text */
            .stTextInput label, .stNumberInput label, .stSelectbox label,
            .stSlider label, .stRadio label, .stCheckbox label {
                color: #000000 !important;
            }

            .stTextInput input, .stNumberInput input, .stSelectbox select {
                color: #000000 !important;
            }

            /* Tab content - ensure black text */
            [data-testid="stTab"] p, [data-testid="stTab"] span,
            [data-testid="stTab"] div, [data-testid="stTab"] li,
            [data-testid="stTab"] strong, [data-testid="stTab"] em {
                color: #000000 !important;
            }

            /* Alert/Info boxes inside tabs */
            .stAlert p, .stAlert span, .stAlert div, .stAlert li {
                color: #000000 !important;
            }

            /* Success/Info/Warning/Error messages */
            [data-testid="stNotification"] p, [data-testid="stNotification"] span,
            [data-testid="stNotification"] div {
                color: #000000 !important;
            }
        </style>
        """

    st.markdown(theme_css, unsafe_allow_html=True)


apply_theme()


if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'tts' not in st.session_state:
    st.session_state.tts = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'last_description' not in st.session_state:
    st.session_state.last_description = ""


@st.cache_resource
def load_detector(model_path: str, conf_threshold: float):
    try:
        detector = YourEyesDetector(model_path=model_path, conf_threshold=conf_threshold)
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_tts(rate: int, volume: float):
    try:
        tts = TextToSpeech(rate=rate, volume=volume)
        return tts
    except Exception as e:
        st.error(f"Error loading TTS: {e}")
        return None


def main():


    st.sidebar.markdown("#  Your Eyes")
    st.sidebar.markdown("### Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📷 Image Mode", "🎥 Video Mode", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Theme")

    theme_col1, theme_col2 = st.sidebar.columns([3, 1])
    with theme_col1:
        current_theme = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Bright Mode"
        st.markdown(f"**Current:** {current_theme}")

    with theme_col2:
        if st.button("🔄", key="theme_toggle", help="Toggle theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.sidebar.markdown("---")

    if page == "🏠 Home":
        show_home_page()
    elif page == "📷 Image Mode":
        show_image_mode()
    elif page == "🎥 Video Mode":
        show_video_mode()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_home_page():

    st.markdown('<div class="main-header"> Your Eyes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Assistive Object Detection for Visually Impaired Users</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    st.markdown("## 🎯 Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is Your Eyes?
        
        **Your Eyes** is an AI-powered assistive technology designed to help visually impaired 
        individuals understand their surroundings through:
        
        - 🔍 **Real-time Object Detection** using YOLO (You Only Look Once)
        - 🔊 **Audio Descriptions** of detected objects and their locations
        - ⚠️ **Priority Alerts** for critical objects (vehicles, traffic signs, people)
        - 📍 **Spatial Awareness** with position and distance information
        
        ### Future Vision
        
        This system is designed to be integrated into **smart glasses** that can:
        - Process video in real-time from a wearable camera
        - Provide continuous audio feedback through bone conduction speakers
        - Operate on low-power embedded devices (Raspberry Pi, Jetson Nano)
        - Work offline for privacy and reliability
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Supported Objects (35 Classes)
        
        Our system can detect and identify:
        
        **🚗 Transportation & Safety:**
        - Vehicles: car, bus, truck, train, motorcycle, bicycle
        - Safety: traffic light, stop sign
        
        **👥 People & Animals:**
        - person, dog, cat
        
        **🏠 Indoor Objects:**
        - Furniture: chair, couch, bed, dining table, bench
        - Electronics: tv, laptop, cell phone
        - Items: cup, bottle, bowl, book, handbag, backpack
        
        **🍽️ Kitchen & Food:**
        - Utensils: knife, fork, spoon
        - Food: sandwich, banana, orange, carrot, apple
        
        **⚽ Sports & Recreation:**
        - sports ball, umbrella
        """)
    
    st.markdown("---")

    st.markdown("## ✨ Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🎯 Smart Detection
        - High accuracy object detection
        - Confidence-based filtering
        - Indoor/Outdoor modes
        """)

    with col2:
        st.markdown("""
        ### 🔊 Audio Feedback
        - Natural voice descriptions
        - Priority object warnings
        - Customizable speech rate
        """)

    with col3:
        st.markdown("""
        ### 📍 Spatial Info
        - Position (left/center/right)
        - Distance estimation
        - Scene understanding
        """)

    st.markdown("---")

    st.markdown("## 🚀 Quick Start")
    st.info("""
    1. **Go to Settings** ⚙️ to configure your model and preferences
    2. **Try Image Mode** 📷 to upload and analyze images
    3. **Use Video Mode** 🎥 for real-time webcam detection
    4. **Listen** 🔊 to audio descriptions of your surroundings
    """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Developed as a student project for assistive technology</p>
        <p>Powered by YOLOv8 • Streamlit • Python</p>
    </div>
    """, unsafe_allow_html=True)


def show_settings_page():

    st.markdown("# ⚙️ Settings")
    st.markdown("Configure your model, TTS, and detection preferences")

    tab1, tab2, tab3 = st.tabs(["🤖 Model Settings", "🔊 TTS Settings", "🎨 Accessibility"])

    with tab1:
        st.markdown("### Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            model_path = st.text_input(
                "Model Path",
                value="yolov8n.pt",
                help="Path to YOLO model weights file"
            )

            st.markdown("**Available Models:**")
            st.markdown("""
            - `yolov8n.pt` - Nano (fastest, least accurate)
            - `yolov8s.pt` - Small
            - `yolov8m.pt` - Medium
            - `yolov8l.pt` - Large
            - `yolov8x.pt` - Extra Large (slowest, most accurate)
            - Custom trained model path
            """)

        with col2:
            default_conf = st.slider(
                "Default Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05
            )

            st.info(f"Current threshold: {default_conf:.2f}")
            st.markdown("""
            **Confidence Threshold Guide:**
            - **0.3-0.4**: More detections, may include false positives
            - **0.5**: Balanced (recommended)
            - **0.6-0.8**: Fewer but more confident detections
            """)

        if st.button("💾 Save Model Settings"):
            st.success("✅ Model settings saved!")

    with tab2:
        st.markdown("### Text-to-Speech Configuration")

        col1, col2 = st.columns(2)

        with col1:
            speech_rate = st.slider(
                "Speech Rate (words per minute)",
                min_value=100,
                max_value=250,
                value=150,
                step=10,
                help="Adjust how fast the voice speaks"
            )

            volume = st.slider(
                "Volume",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1
            )

        with col2:
            description_style = st.radio(
                "Default Description Style",
                ["Simple", "Detailed"],
                help="Choose default description complexity"
            )

            priority_alerts = st.checkbox(
                "Enable Priority Alerts",
                value=True,
                help="Announce critical objects (cars, people, etc.) first"
            )

        st.markdown("### Test TTS")
        test_text = st.text_input(
            "Test Text",
            value="Hello, this is Your Eyes assistive vision system."
        )

        if st.button("🔊 Test Speech"):
            if st.session_state.tts is None:
                st.session_state.tts = load_tts(rate=speech_rate, volume=volume)
            else:
                st.session_state.tts.set_rate(speech_rate)
                st.session_state.tts.set_volume(volume)

            with st.spinner("Speaking..."):
                st.session_state.tts.speak(test_text, blocking=True)
            st.success("✅ Test complete!")

        if st.button("💾 Save TTS Settings"):
            st.success("✅ TTS settings saved!")

    with tab3:
        st.markdown("### Accessibility Options")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Visual Settings")

            high_contrast = st.checkbox(
                "High Contrast Mode",
                help="Increase contrast for better visibility"
            )

            large_text = st.checkbox(
                "Large Text Mode",
                help="Increase font sizes"
            )

            color_scheme = st.selectbox(
                "Color Scheme",
                ["Default", "High Contrast", "Dark Mode", "Light Mode"]
            )

        with col2:
            st.markdown("#### Audio Settings")

            auto_speak = st.checkbox(
                "Auto-speak on Detection",
                value=False,
                help="Automatically speak descriptions when objects are detected"
            )

            keyboard_shortcuts = st.checkbox(
                "Enable Keyboard Shortcuts",
                value=True,
                help="Use keyboard for quick actions"
            )

            st.markdown("""
            **Keyboard Shortcuts:**
            - `Space`: Speak last description
            - `R`: Reload model
            - `C`: Clear results
            """)

        if st.button("💾 Save Accessibility Settings"):
            st.success("✅ Accessibility settings saved!")

    st.markdown("---")
    st.markdown("### 📊 System Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Status", "✅ Loaded" if st.session_state.model_loaded else "❌ Not Loaded")

    with col2:
        st.metric("TTS Status", "✅ Ready" if st.session_state.tts else "❌ Not Ready")

    with col3:
        st.metric("Supported Classes", "35")


def show_video_mode():

    st.markdown("# 🎥 Video Mode")
    st.markdown("Real-time object detection from webcam or video file")

    st.info("""
    **Note:** For the best real-time video experience, this mode uses OpenCV in a separate window.

    **Instructions:**
    1. Select your video source below
    2. Click 'Start Detection'
    3. A new window will open showing live detections
    4. Press 'Q' in the video window to stop
    """)

    st.sidebar.markdown("### Video Mode Settings")

    video_source = st.sidebar.radio(
        "Video Source",
        ["Webcam", "Upload Video File"]
    )

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    detection_mode = st.sidebar.selectbox(
        "Detection Mode",
        ["All Objects", "Indoor Mode", "Outdoor Mode"]
    )

    mode_map = {
        "All Objects": "all",
        "Indoor Mode": "indoor",
        "Outdoor Mode": "outdoor"
    }

    audio_interval = st.sidebar.slider(
        "Audio Description Interval (seconds)",
        min_value=1,
        max_value=10,
        value=3,
        help="How often to speak descriptions during video"
    )

    enable_audio = st.sidebar.checkbox(
        "Enable Audio Descriptions",
        value=True
    )

    model_path = st.sidebar.text_input(
        "Model Path",
        value="yolov8n.pt"
    )

    if st.sidebar.button("🔄 Load/Reload Model"):
        with st.spinner("Loading model..."):
            st.session_state.detector = load_detector(model_path, conf_threshold)
            st.session_state.tts = load_tts(rate=150, volume=1.0)
            if st.session_state.detector:
                st.sidebar.success("✅ Model loaded!")
                st.session_state.model_loaded = True

    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first using the sidebar")
        return

    # Video source selection
    if video_source == "Webcam":
        st.markdown("### 📹 Webcam Detection")

        camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=5,
            value=0,
            help="Usually 0 for built-in webcam, 1+ for external cameras"
        )

        if st.button("▶️ Start Webcam Detection", use_container_width=True):
            st.info("🎥 Opening webcam in new window... Press 'Q' to stop")

            # Run webcam detection
            run_video_detection(
                source=camera_index,
                conf_threshold=conf_threshold,
                mode=mode_map[detection_mode],
                audio_interval=audio_interval,
                enable_audio=enable_audio
            )

            st.success("✅ Webcam detection stopped")

    else:  # Upload Video File
        st.markdown("### 📁 Video File Detection")

        uploaded_video = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv"]
        )

        if uploaded_video is not None:
            # Save uploaded video temporarily
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            if st.button("▶️ Start Video Detection", use_container_width=True):
                st.info("🎥 Processing video in new window... Press 'Q' to stop")

                # Run video detection
                run_video_detection(
                    source=video_path,
                    conf_threshold=conf_threshold,
                    mode=mode_map[detection_mode],
                    audio_interval=audio_interval,
                    enable_audio=enable_audio
                )

                st.success("✅ Video detection completed")


def run_video_detection(
    source,
    conf_threshold: float,
    mode: str,
    audio_interval: int,
    enable_audio: bool
):
    """Run video detection in OpenCV window"""

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.error("❌ Could not open video source")
        return

    last_audio_time = time.time()

    st.write("Video window opened. Press 'Q' in the video window to stop.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process frame
        detected_objects, annotated_frame = st.session_state.detector.process_image(
            frame,
            conf_threshold=conf_threshold,
            mode=mode
        )

        # Add info overlay
        cv2.putText(
            annotated_frame,
            f"Objects: {len(detected_objects)} | Press Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Show frame
        cv2.imshow("Your Eyes - Live Detection", annotated_frame)

        # Audio description at intervals
        if enable_audio and detected_objects:
            current_time = time.time()
            if current_time - last_audio_time >= audio_interval:
                height, width = frame.shape[:2]
                description = st.session_state.tts.generate_detailed_description(
                    detected_objects,
                    image_width=width,
                    image_height=height
                )
                st.session_state.tts.speak(description, blocking=False)
                last_audio_time = current_time

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_image_mode():
    """Display image mode page"""

    st.markdown("# 📷 Image Mode")
    st.markdown("Upload an image to detect objects and hear audio descriptions")

    # Settings in sidebar
    st.sidebar.markdown("### Image Mode Settings")

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for detections"
    )

    detection_mode = st.sidebar.selectbox(
        "Detection Mode",
        ["All Objects", "Indoor Mode", "Outdoor Mode"],
        help="Filter objects based on environment"
    )

    mode_map = {
        "All Objects": "all",
        "Indoor Mode": "indoor",
        "Outdoor Mode": "outdoor"
    }

    description_type = st.sidebar.radio(
        "Description Type",
        ["Simple", "Detailed (with position & distance)"]
    )

    # Model path
    model_path = st.sidebar.text_input(
        "Model Path",
        value="yolov8n.pt",
        help="Path to YOLO model weights"
    )

    # Load detector
    if st.sidebar.button("🔄 Load/Reload Model"):
        with st.spinner("Loading model..."):
            st.session_state.detector = load_detector(model_path, conf_threshold)
            st.session_state.tts = load_tts(rate=150, volume=1.0)
            if st.session_state.detector:
                st.sidebar.success("✅ Model loaded!")
                st.session_state.model_loaded = True

    # Main content
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first using the sidebar")
        st.info("💡 Click '🔄 Load/Reload Model' in the sidebar to get started")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image for object detection"
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display original image
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(image_rgb, use_container_width=True)

        # Process image
        with st.spinner("🔍 Detecting objects..."):
            detected_objects, annotated_image = st.session_state.detector.process_image(
                image,
                conf_threshold=conf_threshold,
                mode=mode_map[detection_mode]
            )
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown("### Detected Objects")
            st.image(annotated_rgb, use_container_width=True)

        # Results section
        st.markdown("---")
        st.markdown("## 📊 Detection Results")

        if detected_objects:
            # Show count
            st.success(f"✅ Found {len(detected_objects)} object(s)")

            # Priority warnings
            priority_objects = [obj for obj in detected_objects if obj.get("is_priority", False)]
            if priority_objects:
                st.markdown('<div class="priority-warning">', unsafe_allow_html=True)
                st.markdown("### ⚠️ Priority Objects Detected!")
                for obj in priority_objects:
                    st.markdown(f"- **{obj['label'].upper()}** (confidence: {obj['confidence']:.2%})")
                st.markdown('</div>', unsafe_allow_html=True)

            # Object list
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### Detected Objects List")
                for i, obj in enumerate(detected_objects, 1):
                    priority_badge = "⚠️" if obj.get("is_priority", False) else "✓"
                    st.markdown(
                        f"{priority_badge} **{i}. {obj['label']}** - "
                        f"Confidence: {obj['confidence']:.2%}"
                    )

            with col2:
                st.markdown("### Statistics")
                from collections import Counter
                labels = [obj["label"] for obj in detected_objects]
                counts = Counter(labels)
                for label, count in counts.most_common():
                    st.metric(label.title(), count)

            # Generate description
            st.markdown("---")
            st.markdown("## 🔊 Audio Description")

            if description_type == "Simple":
                description = st.session_state.tts.generate_simple_description(detected_objects)
            else:
                height, width = image.shape[:2]
                description = st.session_state.tts.generate_detailed_description(
                    detected_objects,
                    image_width=width,
                    image_height=height,
                    include_distance=True,
                    include_position=True
                )

            st.session_state.last_description = description

            st.info(f"📝 **Description:** {description}")

            # Audio controls
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("🔊 Read Description", use_container_width=True):
                    with st.spinner("Speaking..."):
                        st.session_state.tts.speak(description, blocking=True)
                    st.success("✅ Done speaking!")

            with col2:
                if st.button("📋 Copy Description", use_container_width=True):
                    st.code(description)
                    st.success("Description displayed above!")

        else:
            st.warning("No objects detected. Try lowering the confidence threshold.")


if __name__ == "__main__":
    main()

