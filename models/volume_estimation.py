import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from skimage import measure
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def estimate_food_volume(image, yolo_model, pixel_to_cm_ratio=0.1, depth_scale_factor=10.0, yolo_results=None):
    """
    Main function to estimate food volume from a single image using instance segmentation.
    Volume is calculated directly from the depth map.

    Args:
        image: Input image as numpy array (BGR format)
        yolo_model: Loaded YOLO model
        pixel_to_cm_ratio: Conversion ratio from pixels to centimeters
        depth_scale_factor: Factor to scale depth values to centimeters
        yolo_results: Optional pre-computed YOLO results to use instead of running the model again

    Returns:
        List of (food_class, volume) tuples and visualization figure
    """
    # Convert to RGB for processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load depth estimation model using our custom approach
    try:
        depth_model, device = load_midas_model()
    except Exception as e:
        logger.error(f"Error loading depth model: {str(e)}")
        return None, None

    # Use provided YOLO results if available, otherwise run the model
    if yolo_results is not None:
        logger.info("Using provided YOLO results for volume estimation")
        results = [yolo_results]  # Wrap in list to match expected format
    else:
        logger.info("No YOLO results provided, running model")
        # Use the same approach as in image_detection.py to avoid the 'Segment' error
        results = yolo_model(rgb_image, verbose=False)

    # Get depth map using our custom approach
    depth_map = get_depth(rgb_image, depth_model, device)

    # Calculate food volumes
    food_volumes = []
    masks_with_volumes = []

    # Parse segmentation results
    try:
        # Check if masks are available
        has_masks = hasattr(results[0], 'masks') and results[0].masks is not None and len(results[0].masks) > 0
        
        if not has_masks:
            logger.info("No masks found in detection results. Falling back to bounding boxes.")
            # Fall back to bounding boxes if masks aren't available
            return process_with_depth_only(results[0], rgb_image, depth_map, pixel_to_cm_ratio, depth_scale_factor)

        # Get masks and classes
        masks = results[0].masks.data.cpu().numpy()

        # Get class information
        if hasattr(results[0].boxes, 'cls') and hasattr(results[0], 'names'):
            class_ids = results[0].boxes.cls.cpu().numpy()
            class_names = [results[0].names[int(id)] for id in class_ids]
            logger.info(f"Detected class names: {class_names}")
        else:
            class_names = ["food"] * len(masks)
            logger.info("No class names found, using 'food' as default")

        # Process each detected food item with its mask
        for i, mask in enumerate(masks):
            # Convert float mask to binary
            binary_mask = (mask > 0.5).astype(np.uint8)

            # Skip if mask is empty
            if np.sum(binary_mask) == 0:
                continue

            food_class = class_names[i]

            # Get bounding box from mask for reference
            props = measure.regionprops(binary_mask)[0]
            y1, x1, y2, x2 = props.bbox

            # Calculate volume based solely on depth map values for the segmented part only
            # Create a copy of the mask with the same shape as depth_map
            full_mask = np.zeros_like(depth_map, dtype=np.uint8)

            # Handle mask dimension mismatch
            if binary_mask.shape != depth_map.shape:
                # Resize binary_mask to match depth_map dimensions
                binary_mask_resized = np.zeros_like(depth_map, dtype=np.uint8)
                # Only use the part of the mask that fits within the dimensions of depth_map
                h, w = min(binary_mask.shape[0], depth_map.shape[0]), min(binary_mask.shape[1], depth_map.shape[1])
                binary_mask_resized[:h, :w] = binary_mask[:h, :w]
                binary_mask = binary_mask_resized

            # Now apply the mask - only consider pixels that are part of the segmentation
            full_mask[binary_mask > 0] = 1

            # Apply mask to depth map - this isolates only the segmented part
            masked_depth = depth_map * full_mask
            valid_depth_values = masked_depth[full_mask > 0]

            if len(valid_depth_values) == 0:
                continue

            # Calculate volume directly from depth map
            # Each pixel represents pixel_to_cm_ratio × pixel_to_cm_ratio × depth_cm volume
            # Convert pixel dimensions to cm
            pixel_area_cm2 = pixel_to_cm_ratio ** 2  # area of one pixel in cm²
            
            # Calculate volume for each pixel and sum them up
            # For each pixel: area (cm²) × depth (scaled to cm) = volume (cm³)
            pixel_volumes = valid_depth_values * pixel_area_cm2 * depth_scale_factor
            volume = np.sum(pixel_volumes)  # Total volume in cm³

            food_volumes.append((food_class, volume))
            masks_with_volumes.append((binary_mask, food_class, volume, (y1, x1, y2, x2)))

    except Exception as e:
        logger.error(f"Error processing segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

    # Create visualization
    fig = visualize_segmentation_results(rgb_image, depth_map, masks_with_volumes)

    return food_volumes, fig


def process_with_depth_only(results, rgb_image, depth_map, pixel_to_cm_ratio, depth_scale_factor):
    """Process with bounding boxes but calculate volume only from depth map"""
    logger.info("Using bounding box masks with depth-based volume calculation...")

    food_volumes = []
    bboxes_with_volumes = []

    try:
        # Get bounding boxes
        if hasattr(results, 'boxes') and hasattr(results.boxes, 'xyxy'):
            bboxes = results.boxes.xyxy.cpu().numpy()
        else:
            logger.error("No bounding boxes found in results")
            return None, None

        # Get class information
        if hasattr(results.boxes, 'cls') and hasattr(results, 'names'):
            class_ids = results.boxes.cls.cpu().numpy()
            class_names = [results.names[int(id)] for id in class_ids]
            logger.info(f"Detected class names (bbox): {class_names}")
        else:
            class_names = ["food"] * len(bboxes)
            logger.info("No class names found in bboxes, using 'food' as default")

        # Process each detected food item
        for i, bbox in enumerate(bboxes):
            try:
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # Ensure coordinates are within image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(rgb_image.shape[1], x2)
                y2 = min(rgb_image.shape[0], y2)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Create a binary mask from the bounding box
                mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1

                # Apply mask to depth map
                masked_depth = depth_map * mask
                valid_depth_values = masked_depth[mask > 0]

                if len(valid_depth_values) == 0:
                    continue

                # Calculate volume directly from depth map
                # Each pixel represents pixel_to_cm_ratio × pixel_to_cm_ratio × depth_cm volume
                # Convert pixel dimensions to cm
                pixel_area_cm2 = pixel_to_cm_ratio ** 2  # area of one pixel in cm²
                
                # Calculate volume for each pixel and sum them up
                # For each pixel: area (cm²) × depth (scaled to cm) = volume (cm³)
                pixel_volumes = valid_depth_values * pixel_area_cm2 * depth_scale_factor
                volume = np.sum(pixel_volumes)  # Total volume in cm³

                food_class = class_names[i]
                food_volumes.append((food_class, volume))
                bboxes_with_volumes.append((bbox, food_class, volume))
            except Exception as e:
                logger.error(f"Error processing bounding box {i}: {str(e)}")
                continue

        # Create visualization
        if bboxes_with_volumes:
            fig = visualize_bbox_results(rgb_image, depth_map, bboxes_with_volumes)
            return food_volumes, fig
        else:
            logger.warning("No valid bounding boxes with volume could be processed")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in process_with_depth_only: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def load_midas_model():
    """Load a MiDaS model using a simpler approach"""
    try:
        import os  # Ensure os is imported within the function scope
        
        # Set torch hub directory to a local path to avoid permission issues
        torch.hub.set_dir(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'torch_hub'))
        
        # Try to load the small model first (faster)
        try:
            logger.info("Attempting to load MiDaS_small model...")
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        except Exception as e:
            logger.warning(f"Could not load MiDaS_small: {str(e)}. Trying DPT_Hybrid...")
            # Fall back to another model if small fails
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Using device: {device}")
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        logger.error(f"Failed to load MiDaS model: {str(e)}")
        raise RuntimeError(f"Could not load depth estimation model: {str(e)}")

def get_depth(rgb_image, model, device):
    """Get depth map using a simplified approach"""
    try:
        # Preprocess the image
        # Convert to tensor and normalize
        # Check torchvision version for antialias parameter compatibility
        import torchvision
        from packaging import version
        
        # Create transform based on torchvision version
        if version.parse(torchvision.__version__) >= version.parse('0.8.0'):
            try:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256, 256), antialias=True),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except TypeError:
                # Fallback if antialias parameter is not supported
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256, 256)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            # Older versions don't support antialias
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Convert numpy image to PIL for transformation
        if isinstance(rgb_image, np.ndarray):
            # Ensure RGB format for PIL
            if rgb_image.shape[2] == 3:  # Already RGB
                pil_image = Image.fromarray(rgb_image)
            else:
                logger.warning(f"Unexpected image shape: {rgb_image.shape}. Converting to RGB.")
                pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = rgb_image

        # Apply transform and add batch dimension
        input_tensor = transform(pil_image).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # Run inference
        with torch.no_grad():
            try:
                prediction = model(input_tensor)
                
                # Handle different model output formats
                if isinstance(prediction, dict):
                    # Some MiDaS models return a dict
                    prediction = prediction['out']
                
                # Ensure prediction is a tensor
                if not isinstance(prediction, torch.Tensor):
                    raise ValueError(f"Unexpected prediction type: {type(prediction)}")
                
                # Interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1) if prediction.dim() == 3 else prediction,
                    size=rgb_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                # Convert to numpy and normalize
                depth_map = prediction.cpu().numpy()
                depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)
                
                return depth_map
                
            except Exception as e:
                logger.error(f"Error during depth prediction: {str(e)}")
                # Return a simple depth map as fallback
                logger.warning("Using fallback depth map")
                return np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32) * 0.5
                
    except Exception as e:
        logger.error(f"Error in get_depth: {str(e)}")
        # Return a simple depth map as fallback
        return np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32) * 0.5

def is_approximately_cylindrical(food_class):
    """Determine if food is approximately cylindrical"""
    cylindrical_foods = [
         'idli', 'medu vada'
    ]
    # Check if any word from cylindrical_foods is in the food_class
    food_class_lower = food_class.lower()
    return any(food in food_class_lower for food in cylindrical_foods)

def is_approximately_spherical(food_class):
    """Determine if food is approximately spherical"""
    spherical_foods = [
        'samosa'
    ]
    # Check if any word from spherical_foods is in the food_class
    food_class_lower = food_class.lower()
    return any(food in food_class_lower for food in spherical_foods)

def is_approximately_flat(food_class):
    """Determine if food is approximately flat"""
    flat_foods = [
        'dosa', 'roti', 'rice bhakri'
    ]
    # Check if any word from flat_foods is in the food_class
    food_class_lower = food_class.lower()
    return any(food in food_class_lower for food in flat_foods)

def visualize_segmentation_results(image, depth_map, masks_with_volumes):
    """Create visualization of results with segmentation masks"""
    try:
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image with masks
        ax1.imshow(image)
        ax1.set_title("Segmentation Masks")
        ax1.axis('off')
        
        # Plot depth map
        depth_vis = ax2.imshow(depth_map, cmap='plasma')
        ax2.set_title("Depth Map")
        ax2.axis('off')
        fig.colorbar(depth_vis, ax=ax2, label='Depth')
        
        # Add masks to the original image with labels
        for mask, food_class, volume, bbox in masks_with_volumes:
            # Get a random color for this mask
            color = np.random.rand(3)
            
            # Create a mask overlay
            mask_img = np.zeros_like(image)
            for c in range(3):
                mask_img[:, :, c] = np.where(mask > 0, int(color[c] * 255), 0)
            
            # Add mask with transparency
            alpha = 0.4
            ax1.imshow(mask_img, alpha=alpha)
            
            # Add label with volume
            y1, x1, y2, x2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Add text with white background for visibility
            text = f"{food_class}: {volume:.1f} cm³"
            logger.info(f"Adding visualization for {food_class} with volume {volume:.1f} cm³")
            ax1.text(center_x, center_y, text, color='black', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, pad=2))
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return buf
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def visualize_bbox_results(image, depth_map, bboxes_with_volumes):
    """Create visualization of results with bounding boxes (fallback)"""
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Display original image with bounding boxes
        ax1.imshow(image)
        ax1.set_title('Food Detection with Volume (cm³)')
        ax1.axis('off')

        # Display depth map
        depth_vis = ax2.imshow(depth_map, cmap='plasma')
        ax2.set_title('Depth Map')
        ax2.axis('off')
        fig.colorbar(depth_vis, ax=ax2, label='Depth')

        # Add bounding boxes and volume labels
        for bbox, food_class, volume in bboxes_with_volumes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            
            # Add text for volume
            text = f"{food_class}: {volume:.1f} cm³"
            logger.info(f"Adding bbox visualization for {food_class} with volume {volume:.1f} cm³")
            ax1.text(x1, y1-5, text, color='black', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, pad=2),
                    ha='left')

        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return buf
    except Exception as e:
        logger.error(f"Error in bbox visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
