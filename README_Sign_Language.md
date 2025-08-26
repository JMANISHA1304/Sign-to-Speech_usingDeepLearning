# 🤟 Sign Language Recognition System

A comprehensive real-time sign language recognition system using deep learning and computer vision. This system can recognize 26 American Sign Language (ASL) letters in real-time through webcam input.

## 🎯 Features

### Core Features
- **Real-time Recognition**: Live webcam feed with instant sign language detection
- **Multi-Model Ensemble**: Uses both MobileNet and ResNet models for improved accuracy
- **3-Second Hold Timer**: Hold gestures for 3 seconds to add letters to text
- **Palm Detection**: Show open palm to add spaces between words
- **Text-to-Speech**: Convert recognized text to spoken audio
- **Voice Input**: Add text through voice recognition
- **Gesture History**: Track and display all recognized gestures

### Advanced Features
- **Modern GUI**: Beautiful, intuitive interface with real-time feedback
- **Progress Tracking**: Visual progress bar for hold time
- **Confidence Display**: Real-time confidence scores for predictions
- **Text Management**: Save, clear, and manage recognized text
- **Keyboard Shortcuts**: Quick access to all features
- **Error Handling**: Robust error handling and graceful degradation

## 📊 Model Performance

- **MobileNet Model**: ~97.3% accuracy
- **ResNet Model**: ~97.7% accuracy
- **Ensemble Model**: ~98.0% accuracy
- **Real-time Processing**: 10 FPS with webcam input

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam
- Microphone (for voice features)
- Speakers (for text-to-speech)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements_sign_language.txt
   ```

3. **Ensure models are in place**:
   - `models/mobilenet_model.keras`
   - `models/resnet_model.keras`
   - `models/label_mapping.json`

4. **Run the application**:
   ```bash
   python predict_sign.py
   ```

## 🎮 How to Use

### Basic Usage
1. **Start Camera**: Click "🎥 Start Camera" button
2. **Show Signs**: Display ASL letters to the camera
3. **Hold Gestures**: Keep the gesture steady for 3 seconds
4. **Add Spaces**: Show open palm to add spaces
5. **View Text**: See recognized text in the right panel

### Keyboard Shortcuts
- **S**: Speak the current text aloud
- **V**: Start voice input mode
- **C**: Clear all text
- **H**: Show gesture history

### Advanced Features
- **Voice Input**: Click "🎤 Voice Input" to add text by speaking
- **Text-to-Speech**: Click "🔊 Speak Text" to hear the text
- **Save Text**: Click "💾 Save Text" to save to file
- **Gesture History**: Click "📊 Gesture History" to view all gestures

## 📁 Project Structure

```
sign_language_recognition/
├── predict_sign.py              # Main application
├── train_model.py               # Training script
├── generate_graphs.py           # Graph generation
├── requirements_sign_language.txt # Dependencies
├── README_Sign_Language.md      # This file
├── models/                      # Trained models
│   ├── mobilenet_model.keras
│   ├── resnet_model.keras
│   └── label_mapping.json
├── results/                     # Training results
├── graphs/                      # Performance graphs
├── temp_data/                   # Data splits
└── dataset/                     # Training dataset
```

## 🔧 Technical Details

### Model Architecture
- **Base Models**: MobileNetV2 and ResNet50V2
- **Transfer Learning**: Pre-trained on ImageNet
- **Fine-tuning**: Custom layers for sign language classification
- **Ensemble**: Average predictions from both models

### Image Processing
- **Input Size**: 160x160 pixels
- **Preprocessing**: RGB conversion, normalization
- **Augmentation**: Rotation, zoom, shear, shifts during training

### Real-time Features
- **Frame Rate**: 10 FPS
- **Latency**: <100ms prediction time
- **Memory**: Optimized for 8GB RAM systems
- **GPU Support**: Automatic GPU detection and optimization

## 🎨 GUI Features

### Left Panel - Camera
- Live webcam feed with overlay information
- Real-time sign detection display
- Confidence score visualization
- Hold time progress bar
- Camera controls

### Right Panel - Text
- Large text display area
- Real-time text updates
- Feature buttons
- Instructions panel
- Keyboard shortcuts

## 🔍 Palm Detection

The system uses computer vision to detect open palms:
- **HSV Color Space**: Skin color detection
- **Contour Analysis**: Shape and area analysis
- **Threshold Tuning**: Adjustable sensitivity
- **Real-time Feedback**: Visual indicators on screen

## 🎤 Speech Features

### Text-to-Speech
- **Engine**: pyttsx3 (cross-platform)
- **Voice Selection**: Automatic voice detection
- **Speed Control**: Adjustable speech rate
- **Volume Control**: Configurable audio levels

### Voice Recognition
- **Engine**: Google Speech Recognition
- **Microphone Input**: Real-time audio capture
- **Timeout Handling**: 5-second listening window
- **Error Recovery**: Graceful error handling

## 📈 Performance Optimization

### Memory Management
- **Image Resizing**: Efficient preprocessing
- **Batch Processing**: Optimized model inference
- **Garbage Collection**: Automatic memory cleanup
- **Threading**: Non-blocking GUI updates

### Speed Optimization
- **Model Quantization**: Reduced model size
- **Parallel Processing**: Multi-threaded operations
- **GPU Acceleration**: Automatic CUDA detection
- **Frame Skipping**: Configurable processing rate

## 🛠️ Troubleshooting

### Common Issues

1. **Camera not working**:
   - Check webcam permissions
   - Ensure no other application is using the camera
   - Try different camera index (0, 1, 2)

2. **Models not loading**:
   - Verify model files exist in `models/` directory
   - Check file permissions
   - Ensure TensorFlow version compatibility

3. **Speech not working**:
   - Install system audio drivers
   - Check microphone permissions
   - Verify pyttsx3 installation

4. **Performance issues**:
   - Reduce camera resolution
   - Close other applications
   - Check available RAM

### Performance Tips
- Use good lighting for better recognition
- Keep hands clearly visible in camera
- Maintain steady hand positions
- Use a clean background

## 🔮 Future Enhancements

### Planned Features
- **Word Recognition**: Full word detection beyond letters
- **Sentence Structure**: Grammar and syntax analysis
- **Multi-Hand Support**: Two-handed gesture recognition
- **Custom Gestures**: User-defined gesture training
- **Mobile App**: Android/iOS companion app
- **Cloud Integration**: Remote processing capabilities

### Technical Improvements
- **Real-time Translation**: Sign to text to speech pipeline
- **Gesture Learning**: Online learning capabilities
- **Performance Metrics**: Real-time accuracy tracking
- **Model Compression**: Smaller, faster models

## 📚 References

- **ASL Dataset**: American Sign Language letter dataset
- **MobileNetV2**: Efficient CNN architecture
- **ResNet50V2**: Deep residual networks
- **OpenCV**: Computer vision library
- **TensorFlow**: Deep learning framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Performance improvements
- Documentation updates

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- ASL dataset contributors
- Open source community
- TensorFlow and OpenCV teams
- Academic research community

---

**Made with ❤️ for the deaf and hard-of-hearing community**
