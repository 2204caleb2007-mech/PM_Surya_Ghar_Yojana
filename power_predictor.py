import torch
import torch.nn as nn
import numpy as np
import logging
from datetime import datetime
import yaml
import os

logger = logging.getLogger(__name__)

class PowerPredictor:
    def __init__(self, model_path=None):
        self.config = self.load_config()
        self.model_path = model_path or self.config['ml_models']['power_predictor']['model_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
    def load_config(self):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        try:
            model = PowerPredictionModel()
            
            # Load weights if .pth file exists
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading model weights from {self.model_path}")
                
                # Map location for CPU/GPU compatibility
                map_location = self.device if torch.cuda.is_available() else 'cpu'
                
                # Load the state dict
                state_dict = torch.load(self.model_path, map_location=map_location)
                
                # Load weights into model
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()  # Set to evaluation mode
                
                logger.info("Model weights loaded successfully")
            else:
                logger.warning(f"No model weights found at {self.model_path}. Using uninitialized model.")
                
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to mock model")
            return self.create_mock_model()
    
    def create_mock_model(self):
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, x):
                return torch.randn(x.shape[0], 3)
                
        return MockModel()
    
    def prepare_features(self, latitude, longitude, additional_data=None):
        """Prepare input features for the model"""
        # Extract features needed for the model (10 features as defined in PowerPredictionModel)
        month = datetime.now().month
        day_of_year = datetime.now().timetuple().tm_yday
        
        # Normalize latitude and longitude (example normalization)
        lat_norm = (latitude + 90) / 180  # Normalize to [0, 1]
        lon_norm = (longitude + 180) / 360  # Normalize to [0, 1]
        
        # Monthly and seasonal features
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        # Default additional features if not provided
        if additional_data is None:
            additional_data = {
                'tilt_angle': 20.0,  # degrees
                'orientation': 180.0,  # degrees (south-facing)
                'area': 20.0,  # m²
                'efficiency': 0.18,  # panel efficiency
                'temperature': 25.0  # °C
            }
        
        # Prepare feature vector (10 features total)
        features = np.array([
            lat_norm,                     # 0: Normalized latitude
            lon_norm,                     # 1: Normalized longitude
            month_sin,                    # 2: Month sine
            month_cos,                    # 3: Month cosine
            day_sin,                      # 4: Day of year sine
            day_cos,                      # 5: Day of year cosine
            additional_data['tilt_angle'] / 90.0,        # 6: Normalized tilt (0-1)
            additional_data['orientation'] / 360.0,      # 7: Normalized orientation
            additional_data['area'] / 100.0,             # 8: Normalized area
            additional_data['efficiency'],               # 9: Panel efficiency
        ], dtype=np.float32)
        
        return features
    
    def postprocess_predictions(self, model_output, latitude, longitude):
        """Convert model outputs to meaningful predictions"""
        # Assuming model outputs 3 values: [capacity_factor, annual_gen_factor, efficiency_factor]
        capacity_factor = float(model_output[0].item())
        annual_factor = float(model_output[1].item())
        efficiency_factor = float(model_output[2].item())
        
        # Base calculations (scaled by model predictions)
        base_capacity = 3.0  # kW
        capacity_kw = base_capacity * (0.8 + 0.4 * capacity_factor)  # Scale to realistic range
        
        # Adjust based on latitude
        lat_factor = 1 + (8 - abs(latitude - 20)) / 100
        capacity_kw *= lat_factor
        
        # Calculate annual generation
        annual_kwh = capacity_kw * 4 * 365 * (0.8 + 0.4 * annual_factor)  # 4 hours peak sun
        
        # Financial estimates (INR)
        estimated_cost = capacity_kw * 70000  # ₹70,000 per kW
        subsidy_available = min(capacity_kw * 30000, 30000)  # Up to ₹30,000 subsidy
        payback_years = estimated_cost / (annual_kwh * 6)  # ₹6 per kWh savings
        
        return {
            'capacity_kw': round(capacity_kw, 2),
            'annual_kwh': round(annual_kwh),
            'estimated_cost': round(estimated_cost),
            'subsidy_available': round(subsidy_available),
            'payback_period': round(max(3.0, min(15.0, payback_years)), 1),  # Clamp to reasonable range
            'model_confidence': round(float(efficiency_factor), 3)
        }
    
    def predict(self, latitude, longitude, additional_data=None):
        """Predict power generation potential"""
        try:
            # Prepare features
            features = self.prepare_features(latitude, longitude, additional_data)
            
            # Check if we have a trained model (not mock)
            if isinstance(self.model, PowerPredictionModel) and self.model_path and os.path.exists(self.model_path):
                # Use neural network model
                with torch.no_grad():
                    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                    model_output = self.model(input_tensor)
                    
                predictions = self.postprocess_predictions(model_output[0], latitude, longitude)
                predictions['prediction_method'] = 'neural_network'
                
            else:
                # Fall back to heuristic method
                logger.info("Using heuristic prediction (no trained model available)")
                predictions = self.heuristic_predict(latitude, longitude)
                predictions['prediction_method'] = 'heuristic'
            
            return predictions
            
        except Exception as e:
            logger.error(f"Power prediction error: {e}")
            return self.generate_mock_predictions()
    
    def heuristic_predict(self, latitude, longitude):
        """Heuristic prediction method (fallback)"""
        # Base capacity based on typical Indian installations
        base_capacity = 3.0  # kW
        
        # Adjust based on latitude (more sun in south)
        lat_factor = 1 + (8 - abs(latitude - 20)) / 100
        
        # Seasonal adjustment
        month = datetime.now().month
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 3) / 12)
        
        # Calculate predictions
        capacity_kw = base_capacity * lat_factor * seasonal_factor
        annual_kwh = capacity_kw * 4 * 365  # 4 hours of peak sun daily
        
        # Financial estimates (INR)
        estimated_cost = capacity_kw * 70000  # ₹70,000 per kW
        subsidy_available = min(capacity_kw * 30000, 30000)  # Up to ₹30,000 subsidy
        payback_years = estimated_cost / (annual_kwh * 6)  # ₹6 per kWh savings
        
        return {
            'capacity_kw': round(capacity_kw, 2),
            'annual_kwh': round(annual_kwh),
            'estimated_cost': round(estimated_cost),
            'subsidy_available': round(subsidy_available),
            'payback_period': round(max(3.0, min(15.0, payback_years)), 1)
        }
    
    def generate_mock_predictions(self):
        import random
        return {
            'capacity_kw': round(random.uniform(1.0, 10.0), 2),
            'annual_kwh': random.randint(1500, 15000),
            'estimated_cost': random.randint(70000, 700000),
            'subsidy_available': random.randint(10000, 30000),
            'payback_period': round(random.uniform(4.0, 8.0), 1),
            'prediction_method': 'mock'
        }
    
    def is_loaded(self):
        return self.model is not None
    
    def save_model_weights(self, save_path=None):
        """Save current model weights to .pth file"""
        if not isinstance(self.model, PowerPredictionModel):
            logger.warning("Cannot save weights: not a PowerPredictionModel instance")
            return False
            
        save_path = save_path or self.model_path
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save state dict
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model weights saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            return False

class PowerPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # capacity_factor, annual_factor, efficiency_factor
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        return self.network(x)

# Utility function to create and save initial model weights
def initialize_model_weights():
    """Create and save initial model weights"""
    model = PowerPredictionModel()
    
    # Save initial weights
    os.makedirs('model_weights', exist_ok=True)
    torch.save(model.state_dict(), 'model_weights/power_predictor.pth')
    
    # Also create a sample config if needed
    sample_config = {
        'ml_models': {
            'power_predictor': {
                'model_path': 'model_weights/power_predictor.pth'
            }
        }
    }
    
    # Save sample config if config.yaml doesn't exist
    if not os.path.exists('config.yaml'):
        import yaml
        with open('config.yaml', 'w') as f:
            yaml.dump(sample_config, f)
    
    print(f"Initial model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Model weights saved to model_weights/power_predictor.pth")
    
    return model

if __name__ == "__main__":
    # Initialize the model when script is run directly
    initialize_model_weights()