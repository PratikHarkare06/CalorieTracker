import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViewDetector:
    def __init__(self):
        try:
            # List of possible model locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), 'final_yolov8.pt'),
                os.path.join(os.path.dirname(__file__), '..', 'final_yolov8.pt'),
                os.path.join(os.getcwd(), 'final_yolov8.pt'),
                os.path.join(os.path.expanduser('~'), 'Downloads', 'final_yolov8.pt')
            ]
            
            # Try each possible path
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                error_msg = (
                    "Model file 'final_yolov8.pt' not found in any of these locations:\n" +
                    "\n".join(possible_paths)
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info(f"Loading model from: {model_path}")
            # Load model with appropriate task
            self.model = YOLO(model_path)
            # Configure the model to use detection mode only
            # This avoids the 'Segment' object has no attribute 'detect' error
            logger.info("Model loaded successfully")
            
            # Store for later use in volume estimation
            self.last_results = None
            self.detected_objects = []
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def detect_food(self, image):
        try:
            # Run YOLO prediction in detection mode only
            # This avoids the 'Segment' object has no attribute 'detect' error
            results = self.model(image, verbose=False)[0]
            
            # Store the raw results for volume estimation
            self.last_results = results
            
            # Process results
            detected_objects = []
            annotated_image = image.copy()
            
            # Extract detections
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results.names[cls]
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add class name and confidence to the bounding box
                label = f"{class_name} {conf:.2f}"
                # Get text size for better positioning
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Draw filled rectangle for text background
                cv2.rectangle(annotated_image, (x1, y1 - 20), (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Put text on the image
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Store detection
                detected_objects.append({
                    'class': class_name,
                    'confidence': conf,
                    'box': [x1, y1, x2, y2],
                    'type': 'detection'
                })
            
            # Process segmentation masks if available
            if hasattr(results, 'masks') and results.masks is not None:
                for i, mask in enumerate(results.masks):
                    if mask is not None:
                        # Get the corresponding box for this mask
                        if i < len(results.boxes):
                            box = results.boxes[i]
                            cls = int(box.cls[0])
                            class_name = results.names[cls]
                            conf = float(box.conf[0])
                        else:
                            class_name = 'unknown'
                            conf = 0.0
                        
                        # Convert mask to numpy array and apply to image
                        if hasattr(mask, 'data'):
                            # For newer versions of ultralytics
                            mask_array = mask.data.cpu().numpy()
                        else:
                            # For older versions
                            mask_array = mask.cpu().numpy()
                        
                        # Ensure mask is in the right format
                        if len(mask_array.shape) == 3:
                            mask_array = mask_array[0]  # Take first mask if multiple
                        
                        # Resize mask to match image dimensions if needed
                        if mask_array.shape[:2] != (image.shape[0], image.shape[1]):
                            mask_array = cv2.resize(
                                mask_array, 
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # Create a colored mask overlay
                        color = [0, 0, 255]  # Red color for the mask
                        colored_mask = np.zeros_like(image, dtype=np.uint8)
                        for c in range(3):
                            colored_mask[:, :, c] = np.where(mask_array > 0.5, color[c], 0)
                        
                        # Apply the mask with transparency
                        alpha = 0.4  # Transparency factor
                        mask_overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
                        
                        # Only apply mask where it exists
                        mask_binary = (mask_array > 0.5).astype(np.uint8)
                        annotated_image = np.where(
                            np.expand_dims(mask_binary, axis=2) > 0,
                            mask_overlay,
                            annotated_image
                        )
                        
                        # Find the centroid of the mask to place the label
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Get the largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            M = cv2.moments(largest_contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # Add class name and confidence to the segmentation
                                label = f"{class_name} {conf:.2f}"
                                # Get text size for better positioning
                                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                
                                # Draw filled rectangle for text background
                                cv2.rectangle(annotated_image, 
                                             (cx - text_width//2 - 5, cy - text_height - 5), 
                                             (cx + text_width//2 + 5, cy + 5), 
                                             (0, 0, 255), -1)
                                
                                # Put text on the image
                                cv2.putText(annotated_image, label, 
                                           (cx - text_width//2, cy),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        
                        # Store segmentation info
                        detected_objects.append({
                            'class': class_name,
                            'confidence': conf,
                            'type': 'segmentation',
                            'mask_shape': mask_array.shape
                        })
            
            return annotated_image, detected_objects
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise