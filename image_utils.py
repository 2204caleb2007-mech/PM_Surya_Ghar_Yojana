import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.quality_threshold = 0.7
        
    def preprocess_image(self, image_path):
        """Preprocess image for ML analysis"""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Sharpen slightly
            img = img.filter(ImageFilter.SHARPEN)
            
            return img
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None
    
    def detect_edges(self, image):
        """Detect edges in image"""
        try:
            # Convert PIL to OpenCV
            open_cv_image = np.array(image)
            
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            return edges
            
        except Exception as e:
            logger.error(f"Edge detection error: {e}")
            return None
    
    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        try:
            open_cv_image = np.array(image)
            if len(open_cv_image.shape) == 3:
                gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = open_cv_image
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            return sharpness
            
        except Exception as e:
            logger.error(f"Sharpness calculation error: {e}")
            return 0
    
    def assess_quality(self, image):
        """Assess image quality for analysis"""
        sharpness = self.calculate_sharpness(image)
        
        quality_score = min(sharpness / 1000, 1.0)  # Normalize
        
        if quality_score > self.quality_threshold:
            return "HIGH", quality_score
        elif quality_score > 0.4:
            return "MEDIUM", quality_score
        else:
            return "LOW", quality_score
    
    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except:
            return ""
    
    def base64_to_image(self, base64_string):
        """Convert base64 string to PIL image"""
        try:
            image_data = base64.b64decode(base64_string)
            return Image.open(BytesIO(image_data))
        except:
            return None
    
    def resize_image(self, image, target_size=(512, 512)):
        """Resize image while maintaining aspect ratio"""
        try:
            return image.resize(target_size, Image.Resampling.LANCZOS)
        except:
            return image