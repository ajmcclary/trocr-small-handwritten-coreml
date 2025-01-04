import os
import sys
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from transformers import TrOCRProcessor
import coremltools as ct

def download_test_image():
    """Download a sample handwritten text image."""
    url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Failed to download test image: {str(e)}")
        sys.exit(1)

def load_and_preprocess_image(image):
    """Load and preprocess an image for model input."""
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        return pixel_values, processor
    except Exception as e:
        print(f"Failed to process image: {str(e)}")
        sys.exit(1)

def test_coreml_model(model_path, image):
    """Test the converted CoreML model with an image."""
    try:
        # Load CoreML model
        model = ct.models.MLModel(model_path)
        
        # Preprocess image
        image = image.convert('RGB')
        image = image.resize((384, 384), Image.Resampling.BILINEAR)
        
        # Prepare input - CoreML expects PIL Image directly
        input_dict = {'pixel_values': image}
        
        # Run inference
        output = model.predict(input_dict)
        
        # Process output - get token sequences directly
        sequences = list(output.values())[0]
        
        # Load tokenizer for decoding
        tokenizer = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten").tokenizer
        
        # Handle potential sequence format issues
        if isinstance(sequences, np.ndarray):
            sequences = torch.from_numpy(sequences)
        if len(sequences.shape) == 3:  # If we get logits instead of token ids
            sequences = sequences.argmax(dim=-1)
            
        text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Detected Text: {text}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        sys.exit(1)

def main():
    # Check if model exists
    model_path = "TrOCR-Handwritten.mlpackage"
    if not os.path.exists(model_path):
        print(f"Error: Could not find converted model at {model_path}")
        print("Please run convert_trocr.py first to convert the model.")
        sys.exit(1)
    
    # Get image
    print("Downloading sample image...")
    image = download_test_image()
    
    # Test the model
    test_coreml_model(model_path, image)

if __name__ == "__main__":
    main()
