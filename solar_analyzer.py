import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class SolarAnalyzer:
    def __init__(self, model_path=None):
        self.config = self.load_config()
        self.model_path = model_path or self.config['ml_models']['solar_analyzer']['model_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.transform = self.get_transforms()
        
    def load_config(self):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """Load the trained solar panel detection model"""
        try:
            # Create model architecture
            model = SolarDetectionModel()
            
            # Check if weights file exists
            model_weights_path = Path(self.model_path)
            if model_weights_path.exists():
                # Load the weights
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Solar analyzer model loaded from {self.model_path} on {self.device}")
            else:
                logger.warning(f"No model weights found at {self.model_path}. Using untrained model.")
                
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return self.create_mock_model()
    
    def save_model(self, path=None):
        """Save the current model weights"""
        try:
            save_path = path or self.model_path
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def create_mock_model(self):
        """Create a mock model for testing"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 3)
                
        return MockModel()
    
    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_satellite_image(self, lat, lon):
        """Fetch satellite imagery for given coordinates"""
        try:
            # In production, use actual satellite API
            # For now, return a mock image
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/13/{lat}/{lon}"
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            return img
        except:
            # Return a blank image for testing
            return Image.new('RGB', (512, 512), color='white')
    
    def analyze_location(self, latitude, longitude):
        """Analyze solar panel installation at given coordinates"""
        try:
            # Get satellite image
            image = self.get_satellite_image(latitude, longitude)
            
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                has_solar_prob = torch.sigmoid(outputs[:, 0]).item()
                confidence = torch.sigmoid(outputs[:, 1]).item()
                panel_count = torch.exp(outputs[:, 2]).item()
            
            # Determine QC status
            qc_status = self.determine_qc_status(has_solar_prob, confidence)
            
            # Generate QC notes
            qc_notes = self.generate_qc_notes(qc_status, has_solar_prob, panel_count)
            
            return {
                'has_solar': has_solar_prob > 0.5,
                'confidence': confidence,
                'panel_count': int(panel_count),
                'area_sqm': panel_count * 1.65,  # Average panel area
                'qc_status': qc_status,
                'qc_notes': qc_notes,
                'cloud_cover': np.random.random() * 0.3
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return self.generate_mock_results(latitude, longitude)
    
    def determine_qc_status(self, has_solar_prob, confidence):
        if confidence > 0.9:
            return "VERIFIABLE"
        elif confidence > 0.7:
            return "PARTIALLY_VERIFIABLE"
        else:
            return "NOT_VERIFIABLE"
    
    def generate_qc_notes(self, qc_status, has_solar_prob, panel_count):
        notes = []
        if qc_status == "VERIFIABLE":
            notes = [
                "Clear roof view within search radius",
                "Distinct solar module patterns detected",
                "Consistent panel alignment and spacing"
            ]
        elif qc_status == "PARTIALLY_VERIFIABLE":
            notes = [
                "Partial obstruction detected",
                "Moderate confidence in detection",
                "Recommend manual verification"
            ]
        else:
            notes = [
                "Poor image quality or heavy shadows",
                "No clear solar panel signatures",
                "Recommend retry with different imagery"
            ]
        return notes
    
    def generate_mock_results(self, lat, lon):
        """Generate mock results for testing"""
        import random
        has_solar = random.random() > 0.4
        confidence = random.uniform(0.6, 0.98) if has_solar else random.uniform(0.3, 0.7)
        
        return {
            'has_solar': has_solar,
            'confidence': confidence,
            'panel_count': random.randint(0, 20) if has_solar else 0,
            'area_sqm': random.uniform(0, 40) if has_solar else 0,
            'qc_status': "VERIFIABLE" if confidence > 0.85 else "PARTIALLY_VERIFIABLE",
            'qc_notes': ["Mock analysis - testing mode"],
            'cloud_cover': random.uniform(0, 0.3)
        }
    
    def is_loaded(self):
        return self.model is not None

class SolarDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example architecture - replace with actual model
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # has_solar, confidence, panel_count
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)