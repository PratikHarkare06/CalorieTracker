import cv2
import numpy as np
from tensorflow.keras.models import load_model

class ViewDetector:
    def __init__(self):
        self.upper_view_model = load_model('models/upper_view_detector.h5')
        self.side_view_model = load_model('models/side_view_detector.h5')
        
    def detect_view(self, image, view_type='upper'):
        """
        Detect objects based on view type (upper or side)
        """
        if view_type == 'upper':
            model = self.upper_view_model
        else:
            model = self.side_view_model
            
        # Preprocess image
        img = cv2.resize(image, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Get predictions
        pred_boxes = model.predict(img)[0]
        
        # Draw bounding box
        height, width = image.shape[:2]
        x1, y1, x2, y2 = pred_boxes * [width, height, width, height]
        
        # Draw rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        return image, pred_boxes

# Usage in your Streamlit pages: