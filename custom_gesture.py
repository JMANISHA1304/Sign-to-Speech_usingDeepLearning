import cv2
import os
import string
from datetime import datetime

# Base folder to store gestures
base_dir = "dataset/train"  #for training

#base_dir = 'dataset/test'   # for testing


# Setup
labels = list(string.ascii_uppercase) + [str(i) for i in range(10)]
capture_flag = False
current_label = None
img_count = 0

# Initialize webcam
cap = cv2.VideoCapture(0)
print("🎥 Press any key (A-Z, 0-9) to start capturing. Press 'ESC' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror-like view
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    # Draw ROI box
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display label info
    if capture_flag:
        cv2.putText(frame, f"Recording [{current_label}] - {img_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Save ROI
        label_path = os.path.join(base_dir, current_label)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
        full_path = os.path.join(label_path, filename)
        cv2.imwrite(full_path, roi)
        img_count += 1

    # Show the frame
    cv2.imshow("Capture Gestures - Press Key", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC to quit
    if key == 27:
        print("❌ Exiting...")
        break

    # If valid key pressed
    elif chr(key).upper() in labels:
        current_label = chr(key).upper()
        capture_flag = True
        img_count = 0
        print(f"✅ Started capturing for label: {current_label}")

    # Press 's' or spacebar to stop capturing
    elif key == ord('s') or key == 32:
        if capture_flag:
            print(f"🛑 Stopped capturing for label: {current_label}")
        capture_flag = False
        current_label = None
        img_count = 0

cap.release()
cv2.destroyAllWindows()
