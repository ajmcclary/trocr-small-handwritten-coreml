#!/bin/bash

# Exit on any error
set -e

echo "Setting up TrOCR CoreML Converter..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check for Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is required but not installed."
    echo "Please install Python 3.11 using:"
    echo "brew install python@3.11"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for macOS
echo "Installing PyTorch for macOS..."
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Convert model
echo "Converting TrOCR model to CoreML..."
python convert_trocr.py

# Run test with sample image
echo "Testing converted model..."
python test_conversion.py

echo "Setup and conversion completed successfully!"
echo "The converted model is saved as 'TrOCR-Handwritten.mlpackage'"
