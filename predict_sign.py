import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import json
import os
from datetime import datetime
import queue
from PIL import Image, ImageTk

class SignLanguageRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='SystemButtonFace')
        
        # Initialize variables
        self.cap = None
        self.is_camera_on = False
        self.current_prediction = ""
        self.prediction_history = []
        self.hold_time = 0
        self.last_prediction = ""
        self.prediction_threshold = 2.0  # 2 seconds hold time
        self.confidence_threshold = 0.7
        self.space_detected = False
        self.text_content = ""
        self.debug_mode = False  # Set to True to see debug visualizations
        self.box_size = 255  # Size of the detection box
        self.auto_track = True  # Auto-track hand position
        self.box_x = 0  # Dynamic box position
        self.box_y = 0  # Dynamic box position
        
        # Tkinter variables for UI controls
        self.auto_track_var = tk.BooleanVar(value=self.auto_track)
        
        # Frame buffer for snapshot export
        self.last_frame_bgr = None
        
        # Status text variable (initialized early so menu actions can use it)
        self.status_var = tk.StringVar(value="Ready")
        
        # Use main-thread frame loop for UI updates
        self.use_main_loop = True
        
        # Debounce state to prevent auto-repeating the same letter
        self.last_committed_char = ""
        self.released_since_last_commit = True
        self.release_required_seconds = 0.6  # must release for 0.6s before same letter can be accepted again
        self.release_timer_start = 0.0
        
        # Initialize Text-to-Speech engine
        try:
            self.tts_engine = pyttsx3.init()
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', max(120, min(190, rate)))
        except Exception as e:
            self.tts_engine = None
            print(f"⚠️ TTS initialization failed: {e}")
        
        # Initialize with "Hello World" text
        self.text_content = "HELLO WORLD"
        
        # Load models
        self.load_models()
        
        # Setup styles and menu
        self.setup_style()
        self.create_menubar()
        
        # Create GUI
        self.create_gui()
        
        # Initialize camera
        self.init_camera()
        
        # If using main-loop, don't start the background prediction thread
        if not self.use_main_loop:
            self.prediction_queue = queue.Queue()
            self.prediction_thread = threading.Thread(target=self.prediction_worker, daemon=True)
            self.prediction_thread.start()
        else:
            self.prediction_queue = None
        
        # Bind keyboard events
        self.root.bind('<Key>', self.handle_key_press)
        
        # Start GUI update
        self.update_gui()

    def load_models(self):
        """Load trained models and label mapping."""
        try:
            # Load the final ensemble model
            self.model = load_model('models/final_sign_model.keras')
            
            # Load label mapping
            with open('models/label_map.json', 'r') as f:
                self.label_mapping = json.load(f)
            
            # Create reverse mapping
            self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
            
            print("✅ Model loaded successfully!")
            print(f"📊 Model supports {len(self.label_mapping)} classes: {list(self.label_mapping.keys())}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            # Create dummy mapping for testing
            self.label_mapping = {chr(i): i-65 for i in range(65, 91)}  # A-Z
            self.idx_to_label = {v: k for k, v in self.label_mapping.items()}



    def setup_style(self):
        """Configure ttk theme and custom widget styles for a professional UI."""
        try:
            # Improve scaling on high DPI displays
            self.root.call('tk', 'scaling', 1.2)
        except Exception:
            pass
        style = ttk.Style(self.root)
        try:
            if 'vista' in style.theme_names():
                style.theme_use('vista')
            elif 'clam' in style.theme_names():
                style.theme_use('clam')
        except Exception:
            pass

        style.configure('Heading.TLabel', font=('Segoe UI', 22, 'bold'))
        style.configure('Section.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Card.TLabelframe', padding=(10, 8))
        style.configure('Card.TLabelframe.Label', font=('Segoe UI', 11, 'bold'))
        style.configure('Status.TLabel', relief='sunken', anchor='w', padding=(8, 2))

    def create_menubar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label='Start Camera', command=lambda: (not self.is_camera_on) and self.toggle_camera())
        file_menu.add_command(label='Stop Camera', command=lambda: self.is_camera_on and self.toggle_camera())
        file_menu.add_separator()
        file_menu.add_command(label='Save Text', command=self.save_text)
        file_menu.add_command(label='Export Snapshot', command=self.export_snapshot)
        file_menu.add_command(label='Export Session Report', command=self.export_session_report)
        file_menu.add_command(label='Clear Text', command=self.clear_text)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.root.quit)
        menubar.add_cascade(label='File', menu=file_menu)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(label='Auto-Track Hand', variable=self.auto_track_var, command=self.toggle_auto_track)
        menubar.add_cascade(label='View', menu=view_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label='About', command=lambda: messagebox.showinfo(
            'About',
            'Sign Language Recognition System\n\nThis application demonstrates real-time sign recognition with a professional UI built using ttk.'
        ))
        menubar.add_cascade(label='Help', menu=help_menu)

        self.root.config(menu=menubar)

    def create_gui(self):
        """Create the main GUI interface with two-tab layout."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Title
        title_frame = ttk.Frame(main_frame, height=60)
        title_frame.pack(fill=tk.X, pady=(0, 16))
        title_frame.pack_propagate(False)

        title_label = ttk.Label(title_frame, text="Sign Language Recognition System", style='Heading.TLabel')
        title_label.pack(expand=True)

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Overview Tab
        overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(overview_tab, text='Overview')
        self.overview_tab = overview_tab
        self.create_overview_tab(overview_tab)

        # Recognition Tab
        recognition_tab = ttk.Frame(self.notebook)
        self.notebook.add(recognition_tab, text='Recognition')
        self.recognition_tab = recognition_tab

        # Recognition tab contents split in two rows
        top_panel = ttk.Frame(recognition_tab)
        top_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        bottom_panel = ttk.Frame(recognition_tab)
        bottom_panel.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        # Top Panel - Camera and Controls
        self.create_camera_panel(top_panel)

        # Bottom Panel - Text and Features
        self.create_text_panel(bottom_panel)

        # Status bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style='Status.TLabel', anchor='w')
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def create_camera_panel(self, parent):
        """Create camera display and control panel."""
        # Main camera container
        camera_container = ttk.Frame(parent)
        camera_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Camera header
        header_label = ttk.Label(camera_container, text="Live Camera Feed", style='Section.TLabel')
        header_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Camera and controls layout
        camera_controls_frame = ttk.Frame(camera_container)
        camera_controls_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Camera
        camera_frame = ttk.Frame(camera_controls_frame)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Video display with border
        video_container = tk.Frame(camera_frame, bd=1, relief=tk.SUNKEN)
        video_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.video_label = tk.Label(video_container, bg='black', width=640, height=480)
        self.video_label.pack(padx=10, pady=10)
        
        # Right side - Controls
        controls_frame = ttk.Frame(camera_controls_frame, width=300)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        controls_frame.pack_propagate(False)
        
        # Control buttons section
        buttons_section = ttk.LabelFrame(controls_frame, text="Controls", style='Card.TLabelframe')
        buttons_section.pack(fill=tk.X, pady=(0, 15))
        
        buttons_label = ttk.Label(buttons_section, text="Camera", style='Section.TLabel')
        buttons_label.pack(pady=(6, 2))
        
        # Camera control buttons
        self.camera_btn = ttk.Button(buttons_section, text="Start Camera", 
                                   command=self.toggle_camera)
        self.camera_btn.pack(pady=5)
        
        # Clear and Save buttons
        clear_btn = ttk.Button(buttons_section, text="Clear Text", 
                             command=self.clear_text)
        clear_btn.pack(pady=5)
        
        save_btn = ttk.Button(buttons_section, text="Save Text", 
                            command=self.save_text)
        save_btn.pack(pady=5)
        
        # Settings section
        settings_section = ttk.LabelFrame(controls_frame, text="Settings", style='Card.TLabelframe')
        settings_section.pack(fill=tk.X, pady=(0, 15))
        
        settings_label = ttk.Label(settings_section, text="Detection Parameters", style='Section.TLabel')
        settings_label.pack(pady=(6, 2))
        
        # Box size control
        box_control_frame = ttk.Frame(settings_section)
        box_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        box_label = ttk.Label(box_control_frame, text="Detection Box Size:")
        box_label.pack(anchor=tk.W)
        
        self.box_size_var = tk.IntVar(value=self.box_size)
        box_slider = ttk.Scale(box_control_frame, from_=200, to=400, orient=tk.HORIZONTAL,
                             variable=self.box_size_var, command=self.update_box_size,
                             length=200)
        box_slider.pack(fill=tk.X, pady=2)
        
        # Hold time control
        hold_time_frame = ttk.Frame(settings_section)
        hold_time_frame.pack(fill=tk.X, padx=10, pady=5)
        
        hold_time_label = ttk.Label(hold_time_frame, text="Hold Time (seconds):")
        hold_time_label.pack(anchor=tk.W)
        
        self.hold_time_var = tk.DoubleVar(value=self.prediction_threshold)
        hold_time_slider = ttk.Scale(hold_time_frame, from_=0.5, to=5.0, orient=tk.HORIZONTAL,
                                   variable=self.hold_time_var, command=self.update_hold_time,
                                   length=200)
        hold_time_slider.pack(fill=tk.X, pady=2)
        
        # Auto-track toggle
        auto_track_frame = ttk.Frame(settings_section)
        auto_track_frame.pack(fill=tk.X, padx=10, pady=5)
        
        auto_track_check = ttk.Checkbutton(auto_track_frame, text="Auto-Track Hand", 
                                         variable=self.auto_track_var, command=self.toggle_auto_track)
        auto_track_check.pack(anchor=tk.W)
        
        # Status section
        status_section = ttk.LabelFrame(controls_frame, text="Status", style='Card.TLabelframe')
        status_section.pack(fill=tk.BOTH, expand=True)
        
        status_label = ttk.Label(status_section, text="Current Prediction", style='Section.TLabel')
        status_label.pack(pady=(6, 2))
        
        # Current prediction label
        self.prediction_label = ttk.Label(status_section, text="Current Sign: None")
        self.prediction_label.pack(pady=5)
        
        # Hold time progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_section, variable=self.progress_var, 
                                           maximum=self.prediction_threshold, length=250)
        self.progress_bar.pack(pady=5)
        
        # Hold time indicator
        self.hold_time_label = ttk.Label(status_section, text=f"Hold Time: {self.prediction_threshold:.1f}s")
        self.hold_time_label.pack()
        
        # Current hold time
        self.hold_label = ttk.Label(status_section, text="Hold Time: 0.0s")
        self.hold_label.pack(pady=5)

    def create_text_panel(self, parent):
        """Create text display and features panel."""
        # Main text container
        text_container = ttk.Frame(parent)
        text_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Text header
        text_header = ttk.Label(text_container, text="Recognized Text", style='Section.TLabel')
        text_header.pack(fill=tk.X, pady=(0, 10))
        
        # Text and info layout
        text_info_frame = ttk.Frame(text_container)
        text_info_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Text display
        text_frame = ttk.Frame(text_info_frame)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Text display with border
        text_display_container = ttk.LabelFrame(text_frame, text="Output", style='Card.TLabelframe')
        text_display_container.pack(fill=tk.BOTH, expand=True)
        
        self.text_display = scrolledtext.ScrolledText(text_display_container, 
                                                     font=('Segoe UI', 12), 
                                                     height=20, width=50,
                                                     wrap=tk.WORD, relief=tk.FLAT)
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize with "Hello World" text
        self.text_display.insert(1.0, self.text_content)
        
        # Right side - Instructions and History
        info_frame = ttk.Frame(text_info_frame, width=350)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        info_frame.pack_propagate(False)
        
        # Instructions section
        instructions_section = ttk.LabelFrame(info_frame, text="Instructions", style='Card.TLabelframe')
        instructions_section.pack(fill=tk.X, pady=(0, 15))
        
        instructions_label = ttk.Label(instructions_section, text="Usage", style='Section.TLabel')
        instructions_label.pack(pady=(6, 2))
        
        instructions = (
            "1. Position your hand within the detection area.\n"
            "2. Hold a gesture steadily to commit a character (0.5–5.0 s).\n"
            "3. Adjust hold time and box size from Settings.\n"
            "4. Enable Auto-Track to follow hand movement.\n"
            "5. Open palm inserts a space."
        )
        
        instruction_text = ttk.Label(instructions_section, text=instructions, 
                                   justify=tk.LEFT, wraplength=320)
        instruction_text.pack(padx=10, pady=10)
        
        # Features section
        features_section = ttk.LabelFrame(info_frame, text="Features", style='Card.TLabelframe')
        features_section.pack(fill=tk.BOTH, expand=True)
        
        features_label = ttk.Label(features_section, text="Tools", style='Section.TLabel')
        features_label.pack(pady=(6, 2))
        
        # Feature buttons
        history_btn = ttk.Button(features_section, text="Gesture History", 
                               command=self.show_gesture_history)
        history_btn.pack(pady=10)
        
        # Keyboard shortcuts info
        shortcuts_text = (
            "Keyboard Shortcuts:\n"
            "• C – Clear text\n"
            "• H – Show history\n"
            "• T – Toggle Auto-Track"
        )
        
        shortcuts_label = ttk.Label(features_section, text=shortcuts_text, 
                                  justify=tk.LEFT)
        shortcuts_label.pack(pady=10)

    def create_overview_tab(self, parent):
        """Create the overview tab with purpose, applications, and method."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Purpose
        purpose_frame = ttk.LabelFrame(container, text="Purpose", style='Card.TLabelframe')
        purpose_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(purpose_frame,
                  text=(
                      "This system enables real-time recognition of sign language gestures from a live camera feed. "
                      "It aims to facilitate accessible communication and support research and education in human–computer interaction."
                  )).pack(anchor='w', padx=12, pady=8)

        # Applications
        apps_frame = ttk.LabelFrame(container, text="Applications", style='Card.TLabelframe')
        apps_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(apps_frame,
                  text=(
                      "- Assistive technologies for accessibility\n"
                      "- Education and training for sign language learners\n"
                      "- Human–computer interaction and multimodal interfaces\n"
                      "- Data collection and analysis for research"
                  ), justify=tk.LEFT).pack(anchor='w', padx=12, pady=8)

        # Method Overview
        method_frame = ttk.LabelFrame(container, text="How It Works", style='Card.TLabelframe')
        method_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        ttk.Label(method_frame,
                  text=(
                      "- A region of interest (detection box) is tracked in the camera frame.\n"
                      "- The region is preprocessed (resize, normalize) and fed to a trained deep model.\n"
                      "- Predictions with confidence above a threshold initiate a hold timer.\n"
                      "- If the gesture is held steadily for the configured duration, the character is committed.\n"
                      "- An open palm is detected as a space to aid word separation."
                  ), justify=tk.LEFT).pack(anchor='w', padx=12, pady=8)

        # Exports
        export_frame = ttk.LabelFrame(container, text="Exports", style='Card.TLabelframe')
        export_frame.pack(fill=tk.X)
        ttk.Label(export_frame, text="Export artifacts for documentation and publication.").pack(anchor='w', padx=12, pady=(8, 0))
        btns = ttk.Frame(export_frame)
        btns.pack(anchor='w', padx=10, pady=8)
        ttk.Button(btns, text="Export Snapshot", command=self.export_snapshot).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Export Session Report", command=self.export_session_report).pack(side=tk.LEFT)

    def init_camera(self):
        """Initialize camera capture."""
        try:
            # Try different camera backends
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow on Windows
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)  # Fallback to default
            if not self.cap.isOpened():
                print("❌ Could not open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test read a frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print("❌ Camera opened but cannot read frames")
                return False
            
            print("✅ Camera initialized successfully!")
            print(f"📹 Camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
            # Create a placeholder image
            self.create_placeholder_image()
            return True
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False

    def create_placeholder_image(self):
        """Create a placeholder image for when camera is not started."""
        # Create a black image with text
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add text
        cv2.putText(placeholder, "Click 'Start Camera' to begin", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, "Camera Ready", 
                   (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Convert to RGB for tkinter
        placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        placeholder_pil = Image.fromarray(placeholder_rgb)
        placeholder_tk = ImageTk.PhotoImage(placeholder_pil)
        
        # Update video display
        self.video_label.configure(image=placeholder_tk)
        self.video_label.image = placeholder_tk

    def toggle_camera(self):
        """Toggle camera on/off."""
        if self.is_camera_on:
            # Stop camera
            self.is_camera_on = False
            self.camera_btn.config(text="Start Camera")
            print("⏹️ Camera stopped")
            self.status_var.set("Camera stopped")
            
            # Show placeholder image
            self.create_placeholder_image()
            
            # Speak out the composed text upon completion
            if self.text_content.strip():
                self.speak_text(self.text_content.strip())
        else:
            # Start camera
            self.is_camera_on = True
            self.camera_btn.config(text="Stop Camera")
            print("🎥 Camera started - showing live feed")
            self.status_var.set("Camera started")
            
            # Test camera read with multiple attempts
            if self.cap and self.cap.isOpened():
                success = False
                for attempt in range(5):
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✅ Camera feed is working (attempt {attempt + 1})")
                        self.status_var.set("Camera feed active")
                        success = True
                        break
                    time.sleep(0.1)
                
                if success:
                    if self.use_main_loop:
                        self.start_camera_loop()
                else:
                    print("❌ Could not read from camera after multiple attempts")
                    self.is_camera_on = False
                    self.camera_btn.config(text="Start Camera")
                    self.status_var.set("Camera error: cannot read frames")

    def start_camera_loop(self):
        """Start periodic frame processing in the Tk main loop."""
        if self.is_camera_on:
            print("🔄 Starting camera loop...")
            self.root.after(10, self.process_frame)

    def process_frame(self):
        """Process one camera frame and update UI (main thread)."""
        if not self.is_camera_on:
            return
            
        if self.cap is None or not self.cap.isOpened():
            print("❌ Camera not available")
            self.is_camera_on = False
            self.camera_btn.config(text="Start Camera")
            return
            
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("❌ Failed to read frame")
                if self.is_camera_on:
                    self.root.after(100, self.process_frame)
                return
                
            # Ensure frame has valid dimensions
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print("❌ Invalid frame dimensions")
                if self.is_camera_on:
                    self.root.after(100, self.process_frame)
                return
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # Set default box position if not set
            if self.box_x == 0 and self.box_y == 0:
                self.box_x = width - self.box_size - 50
                self.box_y = height - self.box_size - 50
            
            if self.auto_track:
                hand_x, hand_y, hand_area = self.detect_hand_center(frame)
                if hand_x is not None and hand_y is not None:
                    target_x = hand_x - self.box_size // 2
                    target_y = hand_y - self.box_size // 2
                    self.box_x = int(0.7 * self.box_x + 0.3 * target_x)
                    self.box_y = int(0.7 * self.box_y + 0.3 * target_y)
                    self.box_x = max(0, min(width - self.box_size, self.box_x))
                    self.box_y = max(0, min(height - self.box_size, self.box_y))
            
            # Extract detection region safely
            y1 = max(0, self.box_y)
            y2 = min(height, self.box_y + self.box_size)
            x1 = max(0, self.box_x)
            x2 = min(width, self.box_x + self.box_size)
            
            if y2 > y1 and x2 > x1:
                detection_region = frame[y1:y2, x1:x2]
                palm_detected = self.detect_palm(detection_region) if detection_region.size > 0 else False
                predicted_sign, confidence = self.predict_sign(detection_region) if detection_region.size > 0 else ("Unknown", 0.0)
            else:
                palm_detected = False
                predicted_sign, confidence = ("Unknown", 0.0)
            
            # Draw detection box
            box_color = (0, 255, 0)  # Green
            if predicted_sign != "Unknown" and confidence > self.confidence_threshold:
                box_color = (0, 255, 255)  # Yellow
            if palm_detected:
                box_color = (0, 0, 255)  # Red
                
            cv2.rectangle(display_frame, (self.box_x, self.box_y), 
                         (self.box_x + self.box_size, self.box_y + self.box_size), 
                         box_color, 3)
            
            # Add labels
            label_text = "Auto-Tracking" if self.auto_track else "Detection Area"
            cv2.putText(display_frame, label_text, (self.box_x, self.box_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            # Top-right professional info panel
            self.draw_info_panel(display_frame, predicted_sign, confidence, palm_detected)
            
            status_text = "Ready"
            status_color = (0, 255, 0)
            if palm_detected:
                status_text = "PALM DETECTED - SPACE"
                status_color = (0, 0, 255)
            elif predicted_sign != "Unknown" and confidence > self.confidence_threshold:
                status_text = f"Detecting: {predicted_sign}"
                status_color = (0, 255, 255)
                
            cv2.putText(display_frame, status_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Buffer frame for exports
            self.last_frame_bgr = display_frame.copy()

            # Convert to Tk image
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update UI
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk
            
            # Handle prediction
            self.handle_prediction(predicted_sign, confidence, palm_detected)
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
        finally:
            # Schedule next frame
            if self.is_camera_on:
                self.root.after(50, self.process_frame)  # 20 FPS

    def preprocess_image(self, image):
        """Preprocess image for model prediction."""
        # Resize to model input size (224x224 instead of 160x160)
        image = cv2.resize(image, (224, 224))
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image

    def predict_sign(self, image):
        """Predict sign language gesture."""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get prediction from the model
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Get predicted class
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Get label
            if predicted_class in self.idx_to_label:
                predicted_label = self.idx_to_label[predicted_class]
            else:
                predicted_label = "Unknown"
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0

    def draw_info_panel(self, frame, predicted_sign, confidence, palm_detected):
        """Draw a professional top-right information panel with class and confidence."""
        try:
            height, width = frame.shape[:2]
            panel_w, panel_h = 300, 110
            margin = 12
            x0 = max(0, width - panel_w - margin)
            y0 = margin

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # Header
            cv2.putText(frame, "Prediction", (x0 + 12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Text content
            class_text = predicted_sign if (predicted_sign != "Unknown") else "N/A"
            conf_text = (f"{confidence * 100:.1f}%" if predicted_sign != "Unknown" else "—")

            cv2.putText(frame, f"Class: {class_text}", (x0 + 12, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            cv2.putText(frame, f"Confidence: {conf_text}", (x0 + 12, y0 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        except Exception:
            # Fail silently to avoid interrupting the main loop
            pass

    def detect_hand_center(self, image):
        """Detect hand center for auto-tracking."""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin color range
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # If area is large enough, likely a hand
                if area > 5000:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    return center_x, center_y, area
            
            return None, None, 0
                
        except Exception as e:
            print(f"❌ Hand center detection error: {e}")
            return None, None, 0

    def detect_palm(self, image):
        """Detect open palm for space input."""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin color range
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # If area is large enough, likely a palm
                if area > 10000:
                    # Simplify contour to reduce self-intersections
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    if self.debug_mode:
                        debug_frame = image.copy()
                        cv2.drawContours(debug_frame, [approx_contour], -1, (0, 255, 0), 3)
                        cv2.imshow('Palm Detection Debug', debug_frame)
                    
                    try:
                        # Check if contour has enough points for convex hull
                        if len(approx_contour) > 3:
                            # Get convex hull
                            hull = cv2.convexHull(approx_contour, returnPoints=False)
                            
                            # Check if hull indices are valid
                            if len(hull) > 3:
                                defects = cv2.convexityDefects(approx_contour, hull)
                                
                                if defects is not None:
                                    # Count deep defects (finger gaps)
                                    finger_count = 0
                                    for i in range(defects.shape[0]):
                                        s, e, f, d = defects[i][0]
                                        start = tuple(approx_contour[s][0])
                                        end = tuple(approx_contour[e][0])
                                        far = tuple(approx_contour[f][0])
                                        
                                        # Calculate angle between vectors
                                        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                                        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                                        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                                        
                                        # Avoid division by zero
                                        if b * c > 0:
                                            angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c))
                                            
                                            # Angle less than 90 degrees is likely finger gap
                                            if angle <= np.pi/2 and d > 10000:
                                                finger_count += 1
                                    
                                    # Return true for 4 or more fingers (including thumb)
                                    return finger_count >= 3
                        
                        # Fallback to simpler area-based detection
                        aspect_ratio = float(mask.shape[1])/mask.shape[0]
                        if 0.5 <= aspect_ratio <= 2.0:
                            return True
                            
                    except Exception as e:
                        # Fallback to simpler area-based detection
                        aspect_ratio = float(mask.shape[1])/mask.shape[0]
                        if 0.5 <= aspect_ratio <= 2.0:
                            return True
                        
            return False
                
        except Exception as e:
            print(f"❌ Palm detection error: {e}")
            return False

    def prediction_worker(self):
        """Worker thread for continuous prediction."""
        while True:
            try:
                if self.is_camera_on and self.cap is not None:
                    ret, frame = self.cap.read()
                    if ret:
                        # Flip frame horizontally
                        frame = cv2.flip(frame, 1)
                        
                        # Create a copy for display
                        display_frame = frame.copy()
                        
                        # Define detection box position
                        height, width = frame.shape[:2]
                        
                        if self.auto_track:
                            # Auto-track hand position
                            hand_x, hand_y, hand_area = self.detect_hand_center(frame)
                            
                            if hand_x is not None and hand_y is not None:
                                # Center box on hand with smooth movement
                                target_x = hand_x - self.box_size // 2
                                target_y = hand_y - self.box_size // 2
                                
                                # Smooth movement (interpolation)
                                if self.box_x == 0 and self.box_y == 0:
                                    # First detection, set directly
                                    self.box_x = target_x
                                    self.box_y = target_y
                                else:
                                    # Smooth interpolation
                                    self.box_x = int(0.7 * self.box_x + 0.3 * target_x)
                                    self.box_y = int(0.7 * self.box_y + 0.3 * target_y)
                                
                                # Keep box within frame bounds
                                self.box_x = max(0, min(width - self.box_size, self.box_x))
                                self.box_y = max(0, min(height - self.box_size, self.box_y))
                            else:
                                # No hand detected, use default position (right bottom)
                                if self.box_x == 0 and self.box_y == 0:
                                    self.box_x = width - self.box_size - 50
                                    self.box_y = height - self.box_size - 50
                        else:
                            # Fixed position (right bottom area)
                            self.box_x = width - self.box_size - 50
                            self.box_y = height - self.box_size - 50
                        
                        # Extract the detection region
                        detection_region = frame[self.box_y:self.box_y + self.box_size, self.box_x:self.box_x + self.box_size]
                        
                        # Detect palm in the detection region
                        palm_detected = self.detect_palm(detection_region)
                        
                        # Predict sign using the detection region
                        predicted_sign, confidence = self.predict_sign(detection_region)
                        
                        # Draw detection box with different colors based on detection status
                        box_color = (0, 255, 0)  # Green by default
                        if predicted_sign != "Unknown" and confidence > self.confidence_threshold:
                            box_color = (0, 255, 255)  # Yellow when detecting
                        if palm_detected:
                            box_color = (0, 0, 255)  # Red when palm detected
                        
                        cv2.rectangle(display_frame, (self.box_x, self.box_y), (self.box_x + self.box_size, self.box_y + self.box_size), box_color, 3)
                        
                        # Add box label
                        label_text = "Auto-Tracking" if self.auto_track else "Detection Area"
                        cv2.putText(display_frame, label_text, (self.box_x, self.box_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                        
                        # Add corner indicators
                        corner_size = 20
                        cv2.line(display_frame, (self.box_x, self.box_y), (self.box_x + corner_size, self.box_y), box_color, 3)
                        cv2.line(display_frame, (self.box_x, self.box_y), (self.box_x, self.box_y + corner_size), box_color, 3)
                        cv2.line(display_frame, (self.box_x + self.box_size, self.box_y), (self.box_x + self.box_size - corner_size, self.box_y), box_color, 3)
                        cv2.line(display_frame, (self.box_x + self.box_size, self.box_y), (self.box_x + self.box_size, self.box_y + corner_size), box_color, 3)
                        cv2.line(display_frame, (self.box_x, self.box_y + self.box_size), (self.box_x + corner_size, self.box_y + self.box_size), box_color, 3)
                        cv2.line(display_frame, (self.box_x, self.box_y + self.box_size), (self.box_x, self.box_y + self.box_size - corner_size), box_color, 3)
                        cv2.line(display_frame, (self.box_x + self.box_size, self.box_y + self.box_size), (self.box_x + self.box_size - corner_size, self.box_y + self.box_size), box_color, 3)
                        cv2.line(display_frame, (self.box_x + self.box_size, self.box_y + self.box_size), (self.box_x + self.box_size, self.box_y + self.box_size - corner_size), box_color, 3)
                        
                        # Top-right professional info panel
                        self.draw_info_panel(display_frame, predicted_sign, confidence, palm_detected)
                        
                        # Draw detection status
                        status_text = "Ready"
                        status_color = (0, 255, 0)
                        if palm_detected:
                            status_text = "PALM DETECTED - SPACE"
                            status_color = (0, 0, 255)
                        elif predicted_sign != "Unknown" and confidence > self.confidence_threshold:
                            status_text = f"Detecting: {predicted_sign}"
                            status_color = (0, 255, 255)
                        
                        cv2.putText(display_frame, status_text, 
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                        
                        # Draw box size info
                        cv2.putText(display_frame, f"Box Size: {self.box_size}px", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw tracking info
                        tracking_text = "AUTO-TRACKING" if self.auto_track else "FIXED POSITION"
                        tracking_color = (0, 255, 255) if self.auto_track else (255, 255, 255)
                        cv2.putText(display_frame, tracking_text, 
                                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_color, 2)
                        
                        # Buffer frame for exports
                        self.last_frame_bgr = display_frame.copy()

                        # Convert frame for tkinter
                        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        frame_tk = ImageTk.PhotoImage(frame_pil)
                        
                        # Update video display
                        self.video_label.configure(image=frame_tk)
                        self.video_label.image = frame_tk
                        
                        # Handle prediction logic
                        self.handle_prediction(predicted_sign, confidence, palm_detected)
                        
            except Exception as e:
                print(f"❌ Camera feed error: {e}")
                
            time.sleep(0.1)  # 10 FPS

    def handle_prediction(self, predicted_sign, confidence, palm_detected):
        """Handle prediction results with hold-time and release debounce."""
        current_time = time.time()

        # Default for GUI when no hold is ongoing
        hold_duration = 0.0

        # 1) Palm → immediate space and mark as released
        if palm_detected:
            if not self.space_detected:
                self.add_to_text(" ")
                self.space_detected = True
            self.current_prediction = ""
            self.hold_time = 0
            self.released_since_last_commit = True
            self.release_timer_start = 0.0

        # 2) Valid letter prediction
        elif predicted_sign != "Unknown" and confidence > self.confidence_threshold:
            self.space_detected = False

            # If same as last committed and not yet released, block re-commit
            if (predicted_sign == self.last_committed_char) and (not self.released_since_last_commit):
                self.current_prediction = ""
                self.hold_time = 0
            else:
                if predicted_sign != self.current_prediction:
                    self.current_prediction = predicted_sign
                    self.hold_time = current_time
                if self.hold_time > 0:
                    hold_duration = current_time - self.hold_time

                if self.current_prediction and self.hold_time > 0 and hold_duration >= self.prediction_threshold:
                    self.add_to_text(self.current_prediction)
                    self.last_committed_char = self.current_prediction
                    self.released_since_last_commit = False
                    self.release_timer_start = 0.0
                    self.current_prediction = ""
                    self.hold_time = 0

        # 3) No clear prediction → run release timer
        else:
            if self.current_prediction:
                self.current_prediction = ""
                self.hold_time = 0
            self.space_detected = False
            if self.last_committed_char:
                if self.release_timer_start == 0.0:
                    self.release_timer_start = current_time
                elif (current_time - self.release_timer_start) >= self.release_required_seconds:
                    self.released_since_last_commit = True
                    self.release_timer_start = 0.0

        # Update GUI elements
        self.prediction_label.config(text=f"Current Sign: {self.current_prediction if self.current_prediction else 'None'}")
        self.progress_var.set(min(hold_duration if self.hold_time > 0 else 0, self.prediction_threshold))
        self.hold_label.config(text=f"Hold Time: {hold_duration:.1f}s" if self.hold_time > 0 else "Hold Time: 0.0s")

    def add_to_text(self, text):
        """Add text to the display."""
        self.text_content += text
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.text_content)
        
        # Add to history
        self.prediction_history.append({
            'text': text,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })

    def handle_key_press(self, event):
        """Handle keyboard shortcuts."""
        key = event.char.lower()
        
        if key == 'c':
            self.clear_text()
        elif key == 'h':
            self.show_gesture_history()
        elif key == 't':
            # Toggle auto-track
            self.auto_track_var.set(not self.auto_track_var.get())
            self.toggle_auto_track()



    def clear_text(self):
        """Clear the text display."""
        self.text_content = ""
        self.text_display.delete(1.0, tk.END)

    def save_text(self):
        """Save text to file."""
        if self.text_content:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sign_language_text_{timestamp}.txt"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.text_content)
                print(f"✅ Text saved to {filename}")
            except Exception as e:
                print(f"❌ Save error: {e}")

    def export_snapshot(self):
        """Export the most recent camera frame as a PNG for publication."""
        if self.last_frame_bgr is None:
            messagebox.showwarning("Export Snapshot", "No frame available. Start the camera first.")
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"snapshot_{timestamp}.png"
            # Convert BGR to RGB for Pillow and save
            rgb = cv2.cvtColor(self.last_frame_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(out_path)
            self.status_var.set(f"Saved snapshot to {out_path}")
        except Exception as e:
            messagebox.showerror("Export Snapshot", f"Failed to save snapshot: {e}")

    def export_session_report(self):
        """Export a concise text report of the session suitable for appendices."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"session_report_{timestamp}.txt"
            lines = []
            lines.append("Sign Language Recognition System - Session Report\n")
            lines.append(f"Timestamp: {datetime.now().isoformat()}\n")
            lines.append(f"Hold time threshold: {self.prediction_threshold:.1f}s\n")
            lines.append(f"Detection box size: {self.box_size}px\n")
            lines.append(f"Auto-Track: {'Enabled' if self.auto_track else 'Disabled'}\n")
            lines.append("\nRecognized Text:\n")
            lines.append(self.text_content + "\n")
            lines.append("\nGesture History (time, token):\n")
            for item in self.prediction_history[-200:]:
                token = item.get('text', '')
                ts = item.get('timestamp', '')
                lines.append(f"{ts}\t{token}\n")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            self.status_var.set(f"Saved session report to {out_path}")
        except Exception as e:
            messagebox.showerror("Export Session Report", f"Failed to save report: {e}")

    def show_gesture_history(self):
        """Show gesture prediction history."""
        if not self.prediction_history:
            return
        
        # Create history window
        history_window = tk.Toplevel(self.root)
        history_window.title("Gesture History")
        history_window.geometry("400x500")
        
        # Create treeview
        tree = ttk.Treeview(history_window, columns=('Time', 'Gesture'), show='headings')
        tree.heading('Time', text='Time')
        tree.heading('Gesture', text='Gesture')
        
        # Add data
        for item in self.prediction_history:
            tree.insert('', 'end', values=(item['timestamp'], item['text']))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_window, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

    def toggle_auto_track(self):
        """Toggle auto-tracking on/off."""
        self.auto_track = self.auto_track_var.get()
        if self.auto_track:
            print("🎯 Auto-Track enabled - following hand position")
        else:
            print("📍 Fixed position mode - detection box in right-bottom")
        
        # Reset box position when toggling
        self.box_x = 0
        self.box_y = 0

    def update_hold_time(self, value):
        """Update the hold time threshold."""
        self.prediction_threshold = float(value)
        self.hold_time_label.config(text=f"Hold Time: {self.prediction_threshold:.1f}s")
        self.progress_bar.config(maximum=self.prediction_threshold)
        print(f"⏱️ Hold time updated to: {self.prediction_threshold:.1f}s")

    def update_box_size(self, value):
        """Update the detection box size."""
        self.box_size = int(value)
        print(f"📏 Detection box size updated to: {self.box_size}px")

    def update_gui(self):
        """Update GUI elements."""
        # This method can be used for periodic GUI updates
        self.root.after(100, self.update_gui)

    def on_closing(self):
        """Handle application closing."""
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def speak_text(self, text):
        """Speak provided text using TTS in a background thread."""
        if not text or self.tts_engine is None:
            return
        
        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"⚠️ TTS error: {e}")
        
        threading.Thread(target=_speak, daemon=True).start()

def main():
    """Main function."""
    root = tk.Tk()
    app = SignLanguageRecognitionApp(root)
    
    # Set closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()
