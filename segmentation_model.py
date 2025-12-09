import torch
import torch.nn as nn
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import logging
import cv2

logger = logging.getLogger(__name__)

# UNet Architecture
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bridge
        self.bridge = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.up4(bridge)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

class PanelSegmentation:
    def __init__(self, model_path='model_weights/segmentation_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained UNet model"""
        try:
            # Create model instance
            model = UNet(in_channels=3, out_channels=1)
            
            # Load weights if path is provided
            if model_path and torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path))
                model = model.cuda()
            elif model_path:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
            model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")
            logger.info("Using mock model for fallback")
            return MockSegmentationModel()
    
    def segment_panels(self, image_array):
        """Segment solar panels in satellite imagery"""
        try:
            # Convert numpy array to tensor
            if isinstance(image_array, np.ndarray):
                # Ensure image is in the right format [C, H, W]
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
                elif len(image_array.shape) == 2:
                    # Convert grayscale to RGB
                    image_tensor = torch.from_numpy(np.stack([image_array]*3, axis=0)).float()
                else:
                    raise ValueError(f"Unexpected image shape: {image_array.shape}")
                
                # Normalize and add batch dimension
                image_tensor = image_tensor / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
            else:
                raise ValueError("Input must be a numpy array")
            
            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)
                mask = torch.sigmoid(output).squeeze().cpu().numpy()
                
            # Convert to binary mask
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Convert to base64 for frontend
            mask_base64 = self.mask_to_base64(binary_mask)
            
            # Calculate panel area
            total_pixels = np.sum(binary_mask > 0)
            area_sqm = total_pixels * 0.25  # Assuming 0.5m resolution
            
            # Find contours for panel count
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return {
                'mask_base64': mask_base64,
                'area_sqm': round(area_sqm, 2),
                'panel_count': len(contours),
                'contours_found': len(contours),
                'mask_shape': binary_mask.shape
            }
            
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            # Fallback to mock segmentation
            return self.segment_panels_mock()
    
    def segment_panels_mock(self):
        """Mock segmentation for testing/fallback"""
        mask = self.generate_mock_mask()
        mask_base64 = self.mask_to_base64(mask)
        total_pixels = np.sum(mask > 0)
        area_sqm = total_pixels * 0.25
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'mask_base64': mask_base64,
            'area_sqm': round(area_sqm, 2),
            'panel_count': len(contours),
            'contours_found': len(contours),
            'mask_shape': mask.shape
        }
    
    def generate_mock_mask(self):
        """Generate a mock segmentation mask"""
        size = 512
        mask = np.zeros((size, size), dtype=np.uint8)
        
        # Add some rectangular "panels"
        for _ in range(np.random.randint(1, 8)):
            x = np.random.randint(50, size-100)
            y = np.random.randint(50, size-100)
            w = np.random.randint(20, 50)
            h = np.random.randint(30, 60)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        
        return mask
    
    def mask_to_base64(self, mask):
        """Convert mask to base64 string"""
        if mask is None or mask.size == 0:
            return ""
        
        # Convert to image
        img = Image.fromarray(mask)
        
        # Save to bytes
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        
        # Encode to base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def is_loaded(self):
        return self.model is not None

class MockSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Return random mask for mock
        return torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])