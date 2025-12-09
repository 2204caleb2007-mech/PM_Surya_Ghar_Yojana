# Solar Panel Analysis System

## Overview
AI-powered solar panel detection and analysis system for PM Surya Ghar Yojana verification.

## Features
1. **AI-Powered Solar Detection**: Detect solar panels from satellite imagery
2. **Power Generation Prediction**: Estimate energy output
3. **Quality Control**: Verify installation quality
4. **3D Visualization**: Interactive 3D satellite view
5. **Batch Processing**: Analyze multiple locations via CSV
6. **Financial Analysis**: Cost, subsidy, and payback period calculations

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for ML models)
- 8GB+ RAM

### Setup
```bash
# Clone repository
git clone https://2204caleb2007-mech.github.io/PMSuryaGharYojana/
cd solar_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights (place in model_weights folder)
# Download from: <model-weights-url>