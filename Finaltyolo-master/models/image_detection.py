import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViewDetector:
    def __init__(self):
        try:
            # Get the absolute path to the models directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'models', 'final_yolo8.pt')
            logger.info(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(
                    f"Model file not found. Please place final_yolo8.pt in: {os.path.join(base_dir, 'models')}"
