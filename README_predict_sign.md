# Sign Language Recognition System

A real-time sign language recognition application that uses computer vision and deep learning to recognize American Sign Language (ASL) gestures.

## Features

- **Real-time Camera Feed**: Live video capture for gesture recognition
- **ASL Alphabet Recognition**: Recognizes all 26 letters (A-Z)
- **Space Detection**: Detects open palm for space input
- **Auto-Tracking**: Automatic hand tracking for better user experience
- **Adjustable Hold Time**: Draggable slider from 0.5s to 5.0s for gesture confirmation
- **Detection Box**: Configurable detection area with visual feedback
- **Gesture History**: Tracks and displays gesture history
- **Text Saving**: Save recognized text to files
- **Modern UI**: Elegant dark theme with organized layout

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the trained model files:
   - `models/final_sign_model.keras` - The trained model
   - `models/label_map.json` - Label mapping file

## Usage

1. Run the application:
```bash
python predict_sign.py
```

2. The GUI will open with the following features:
   - **Left Panel**: Camera feed and controls
   - **Right Panel**: Recognized text and features

3. **Getting Started**:
   - Click "🎥 Start Camera" to begin
   - Position your hand in the green detection box
   - Show ASL alphabet gestures
   - Hold gestures for adjustable time (0.5s to 5.0s) to add letters
   - Enable Auto-Track for automatic hand following

## Controls

### Camera Controls
- **Start Camera**: Begin live video capture
- **Stop Camera**: Stop video capture
- **Clear Text**: Clear the recognized text
- **Save Text**: Save text to a file

### Keyboard Shortcuts
- **C**: Clear the text
- **H**: Show gesture history
- **T**: Toggle Auto-Track mode

### Gesture Recognition
- **Hold Gesture**: Hold a sign language gesture for adjustable time (0.5s to 5.0s) to add it to text
- **Open Palm**: Show an open palm to add a space
- **Confidence**: Only gestures with >70% confidence are recognized
- **Auto-Track**: Automatically follows your hand position for better user experience
- **Detection Box**: Visual feedback with color-coded status (Green=Ready, Yellow=Detecting, Red=Palm)

## How It Works

1. **Image Preprocessing**: Captures frames from camera and preprocesses them
2. **Gesture Recognition**: Uses a trained CNN model to classify ASL gestures
3. **Palm Detection**: Detects open palms for space input using contour analysis
4. **Hold Time Logic**: Requires holding gestures for 3 seconds to confirm
5. **Text Generation**: Builds text from recognized gestures

## Model Information

- **Architecture**: CNN-based ensemble model
- **Input Size**: 224x224 pixels
- **Classes**: 26 (A-Z)
- **Accuracy**: High accuracy on trained dataset

## Troubleshooting

### Common Issues

1. **Camera not working**: Make sure your webcam is connected and not in use by another application
2. **Model not found**: Ensure the model files are in the `models/` directory
3. **Performance issues**: Close other applications to free up system resources
4. **Auto-tracking issues**: Ensure good lighting for hand detection

### Error Messages

- **"Could not open camera"**: Check camera connection and permissions
- **"Model loading failed"**: Verify model files exist and are not corrupted


## File Structure

```
├── predict_sign.py          # Main application file
├── requirements.txt         # Python dependencies
├── models/
│   ├── final_sign_model.keras  # Trained model
│   └── label_map.json          # Label mapping
└── README_predict_sign.md   # This file
```

## System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.7 or higher
- **Camera**: Webcam or USB camera
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space

## License

This project is for educational and research purposes.
