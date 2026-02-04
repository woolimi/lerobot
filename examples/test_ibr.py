
import cv2
import numpy as np
import sys
import os

# Ensure lerobot is in path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_path)
print(f"Added {src_path} to sys.path")
import lerobot
print(f"Lerobot location: {os.path.dirname(lerobot.__file__)}")

from lerobot.utils.ibr import IBRModule

def test_ibr():
    print("Initializing IBRModule...")
    # Initialize with default model (will download if needed)
    ibr = IBRModule(model_path="yolov8n-seg.pt", conf=0.5)

    print("Creating dummy image...")
    # Create a dummy image (e.g., a green circle on a blue background)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (255, 0, 0) # Blue background
    cv2.circle(image, (320, 240), 100, (0, 255, 0), -1) # Green circle (dummy object)
    
    # Note: YOLO might not detect this synthetic shape as a known class, 
    # but the test ensures the pipeline runs without crashing.
    # To really test detection, we'd need a real image or a better synthetic one.
    # For now, we just check if it runs.

    print("Processing image...")
    output = ibr.process(image)

    print(f"Input shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    
    assert image.shape == output.shape, "Output shape mismatch"
    assert output.dtype == np.uint8, "Output dtype mismatch"

    print("Test passed! (Note: Object detection might fail on synthetic image, but pipeline is functional)")
    
    # Optional: Save output
    cv2.imwrite("output_ibr.jpg", output)
    print("Saved output to output_ibr.jpg")

if __name__ == "__main__":
    test_ibr()
