import cv2
import numpy as np

def test_detection_box():
    """Test the detection box functionality."""
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Define detection box (right bottom area)
    height, width = test_image.shape[:2]
    box_size = 300
    box_x = width - box_size - 50  # 50 pixels from right edge
    box_y = height - box_size - 50  # 50 pixels from bottom edge
    
    # Draw detection box
    cv2.rectangle(test_image, (box_x, box_y), (box_x + box_size, box_y + box_size), 
                 (0, 255, 0), 3)
    cv2.putText(test_image, "Detection Area", (box_x, box_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add corner indicators
    corner_size = 20
    cv2.line(test_image, (box_x, box_y), (box_x + corner_size, box_y), (0, 255, 0), 3)
    cv2.line(test_image, (box_x, box_y), (box_x, box_y + corner_size), (0, 255, 0), 3)
    cv2.line(test_image, (box_x + box_size, box_y), (box_x + box_size - corner_size, box_y), (0, 255, 0), 3)
    cv2.line(test_image, (box_x + box_size, box_y), (box_x + box_size, box_y + corner_size), (0, 255, 0), 3)
    cv2.line(test_image, (box_x, box_y + box_size), (box_x + corner_size, box_y + box_size), (0, 255, 0), 3)
    cv2.line(test_image, (box_x, box_y + box_size), (box_x, box_y + box_size - corner_size), (0, 255, 0), 3)
    cv2.line(test_image, (box_x + box_size, box_y + box_size), (box_x + box_size - corner_size, box_y + box_size), (0, 255, 0), 3)
    cv2.line(test_image, (box_x + box_size, box_y + box_size), (box_x + box_size, box_y + box_size - corner_size), (0, 255, 0), 3)
    
    # Add instructions
    cv2.putText(test_image, "Detection Box Test", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_image, "Green box shows detection area", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(test_image, "Position your hand in this area", (10, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the test image
    cv2.imshow('Detection Box Test', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ Detection box test completed!")
    print(f"📏 Box size: {box_size}px")
    print(f"📍 Box position: ({box_x}, {box_y})")
    print(f"📐 Box dimensions: {box_size}x{box_size}")

if __name__ == "__main__":
    test_detection_box()

