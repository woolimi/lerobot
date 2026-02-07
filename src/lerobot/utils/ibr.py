import cv2
import numpy as np
import torch
from ultralytics import YOLO

class IBRModule:
    """
    Image Background Removal (IBR) module using YOLOv8 Segmentation.
    """
    def __init__(self, model_path: str = "yolov8n-seg.pt", conf: float = 0.5, target_classes: list[str] | None = None):
        """
        Initialize the IBR module.

        Args:
            model_path (str): Path to the YOLOv8 segmentation model.
            conf (float): Confidence threshold for detection.
            target_classes (list[str] | None): List of class names to keep. If None, keeps all detected objects.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.target_classes = target_classes
        
        # Warmup
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image to remove background.

        Args:
            image (np.ndarray): Input image (H, W, C) in BGR or RGB format.
        
        Returns:
            np.ndarray: Image with background removed (pixels set to 0).
        """
        # Run inference
        results = self.model(image, conf=self.conf, verbose=False)
        
        if not results or results[0].masks is None:
            # Return black image if no object detected
            return np.zeros_like(image)

        result = results[0]
        masks = result.masks.data  # shape (N, H, W)
        boxes = result.boxes
        
        # Resize masks to original image size
        # YOLOv8 masks might be smaller than original image size
        if masks.shape[1:] != image.shape[:2]:
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1), 
                size=image.shape[:2], 
                mode="bilinear", 
                align_corners=False
            ).squeeze(1)

        final_mask = torch.zeros(image.shape[:2], device=masks.device, dtype=torch.bool)

        # Filter by class if specified
        if self.target_classes:
            target_indices = []
            for i, cls_id in enumerate(boxes.cls):
                cls_name = result.names[int(cls_id)]
                if cls_name in self.target_classes:
                    target_indices.append(i)
            
            if not target_indices:
                return np.zeros_like(image)
            
            for idx in target_indices:
                final_mask = torch.logical_or(final_mask, masks[idx] > 0.5)
        else:
             # Combine all masks
            for i in range(len(masks)):
                final_mask = torch.logical_or(final_mask, masks[i] > 0.5)

        # Convert back to numpy uint8
        final_mask_np = final_mask.cpu().numpy().astype(np.uint8) * 255
        
        # Apply mask
        output_image = cv2.bitwise_and(image, image, mask=final_mask_np)
        
        return output_image
