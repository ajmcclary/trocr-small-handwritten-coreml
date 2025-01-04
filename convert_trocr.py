import os
import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import coremltools as ct

def load_pytorch_model():
    """Load the TrOCR model and processor from HuggingFace."""
    try:
        print("Loading TrOCR model and processor...")
        
        from huggingface_hub import snapshot_download
        
        # Download model files
        model_path = snapshot_download(
            repo_id="microsoft/trocr-small-handwritten",
            allow_patterns=["*.json", "*.bin", "*.txt", "*.model", "*.safetensors"],
            local_files_only=False,
            ignore_patterns=[".*"]
        )
        
        # Load model from downloaded files
        model = VisionEncoderDecoderModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_safetensors=False  # Try PyTorch bin file instead
        )
        
        # Load processor components separately
        from transformers import ViTImageProcessor, XLMRobertaTokenizer
        
        image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/trocr-small-handwritten")
        
        # Combine into processor
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        
        return model, processor
    except Exception as e:
        raise Exception(f"Failed to load PyTorch model: {str(e)}")

def preprocess_image(image_path, processor):
    """Preprocess an image for model input."""
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        return pixel_values
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {str(e)}")

def convert_to_coreml(model, processor):
    """Convert PyTorch model to CoreML format."""
    try:
        from transformers import XLMRobertaTokenizer
        
        print("Preparing model for conversion...")
        model.eval()
        
        # Create dummy inputs
        dummy_pixel_values = torch.randn(1, 3, 384, 384)
        
        # Create a simplified forward function for tracing
        class SimpleWrapper(torch.nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.register_buffer('start_token', torch.tensor([[2]]))  # decoder_start_token_id
            
            def forward(self, pixel_values):
                # Get encoder outputs
                encoder_outputs = self.encoder(pixel_values.contiguous(), return_dict=False)[0]
                
                # Initialize sequence with start token
                current_ids = self.start_token
                max_length = 20  # Increased maximum length
                
                # Initialize sequence generation
                generated_tokens = [current_ids]
                last_tokens = []  # Track recent tokens for repetition penalty
                
                # Generate sequence
                for _ in range(max_length - 1):
                    # Get decoder outputs
                    decoder_outputs = self.decoder(
                        input_ids=current_ids,
                        encoder_hidden_states=encoder_outputs,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    # Get logits for next token
                    logits = decoder_outputs.logits[:, -1, :]
                    
                    # Apply temperature scaling and repetition penalty
                    temperature = 0.3  # Lower temperature for more focused sampling
                    scaled_logits = logits / temperature
                    
                    # Apply repetition penalty
                    if len(last_tokens) > 0:
                        for prev_token in last_tokens[-3:]:  # Look at last 3 tokens
                            scaled_logits[0, prev_token] /= 2.0  # Penalize recent tokens
                    
                    # Get probabilities and next token
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                    
                    # Update last tokens
                    last_tokens.append(next_token.item())
                    
                    # Stop conditions
                    if next_token.item() == 2:  # eos_token_id
                        break
                        
                    # Get current sequence
                    current_text = current_ids[0].tolist()
                    
                    # Stop if sequence is too long
                    if len(current_text) >= 20:  # Increased maximum length
                        break
                        
                    # Stop if we detect any repetition
                    if len(current_text) >= 6:  # Increased pattern length
                        last_three = current_text[-3:]
                        prev_three = current_text[-6:-3]
                        if last_three == prev_three:
                            break
                    
                    # Add the predicted token to the sequence
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    generated_tokens.append(next_token)
                
                # Return the generated sequence
                return current_ids
        
        # Create and trace the simplified model
        wrapped_model = SimpleWrapper(model.encoder, model.decoder)
        wrapped_model.eval()
        
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapped_model, dummy_pixel_values)
        
        # Convert to CoreML with classifier config
        print("Converting to CoreML...")
        
        # Get tokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/trocr-small-handwritten")
        
        # Create a custom output processing function
        def process_output(logits):
            # Convert logits to token ids
            token_ids = torch.argmax(torch.tensor(logits), dim=-1)
            # Decode tokens to text
            text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
            return text
        
        # Convert to CoreML without classifier config
        print("Converting to CoreML...")
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.ImageType(
                    name="pixel_values",
                    shape=dummy_pixel_values.shape,
                    color_layout=ct.colorlayout.RGB,
                    scale=1.0/255.0  # Normalize pixel values
                )
            ],
            minimum_deployment_target=ct.target.macOS13,
            compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine when available
            convert_to="mlprogram"  # Use the new backend
        )
        
        # Set model metadata
        mlmodel.author = "Converted from microsoft/trocr-small-handwritten"
        mlmodel.license = "MIT"
        mlmodel.short_description = "TrOCR model for handwritten text recognition"
        mlmodel.version = "1.0"
        
        return mlmodel
    except Exception as e:
        raise Exception(f"Failed to convert model: {str(e)}")

def validate_conversion(pytorch_model, coreml_model, processor, test_image_path):
    """Validate the converted model produces similar results to PyTorch model."""
    try:
        # Load tokenizer for decoding
        tokenizer = processor.tokenizer
        
        # PyTorch prediction
        pytorch_input = preprocess_image(test_image_path, processor)
        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs = pytorch_model.encoder(pytorch_input, return_dict=False)[0]
            # Initialize decoder input
            current_ids = torch.tensor([[pytorch_model.config.decoder.decoder_start_token_id]])
            
            # Initialize generation
            last_tokens = []  # Track recent tokens for repetition penalty
            
            # Generate sequence
            for _ in range(20 - 1):  # max_length - 1
                # Get decoder outputs
                decoder_outputs = pytorch_model.decoder(
                    input_ids=current_ids,
                    encoder_hidden_states=encoder_outputs,
                    use_cache=True,
                    return_dict=True
                )
                
                # Get logits for next token
                logits = decoder_outputs.logits[:, -1, :]
                
                # Apply temperature scaling and repetition penalty
                temperature = 0.3  # Lower temperature for more focused sampling
                scaled_logits = logits / temperature
                
                # Apply repetition penalty
                if len(last_tokens) > 0:
                    for prev_token in last_tokens[-3:]:  # Look at last 3 tokens
                        scaled_logits[0, prev_token] /= 2.0  # Penalize recent tokens
                
                # Get probabilities and next token
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Update last tokens
                last_tokens.append(next_token.item())
                
                # Stop conditions
                if next_token.item() == 2:  # eos_token_id
                    break
                    
                # Get current sequence
                current_text = current_ids[0].tolist()
                
                # Stop if sequence is too long
                if len(current_text) >= 20:  # Increased maximum length
                    break
                    
                # Stop if we detect any repetition
                if len(current_text) >= 6:  # Increased pattern length
                    last_three = current_text[-3:]
                    prev_three = current_text[-6:-3]
                    if last_three == prev_three:
                        break
                
                # Add the predicted token to the sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
            
            pytorch_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        
        # CoreML prediction
        coreml_input = {'pixel_values': pytorch_input.numpy()}
        coreml_output = coreml_model.predict(coreml_input)
        token_ids = list(coreml_output.values())[0][0]
        coreml_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print("\nValidation Results:")
        print(f"PyTorch output: {pytorch_text}")
        print(f"CoreML output: {coreml_text}")
        
        return pytorch_text, list(coreml_output.values())[0]
    except Exception as e:
        raise Exception(f"Validation failed: {str(e)}")

def main():
    try:
        # Load PyTorch model
        model, processor = load_pytorch_model()
        
        # Convert to CoreML
        coreml_model = convert_to_coreml(model, processor)
        
        # Save the model
        output_path = "TrOCR-Handwritten.mlpackage"
        print(f"\nSaving CoreML model to {output_path}")
        coreml_model.save(output_path)
        
        # Validate if test image is provided
        if len(sys.argv) > 1:
            test_image_path = sys.argv[1]
            if os.path.exists(test_image_path):
                validate_conversion(model, coreml_model, processor, test_image_path)
            else:
                print(f"Test image not found: {test_image_path}")
        
        print("\nConversion completed successfully!")
        print("\nNotes:")
        print("- The converted model requires macOS 13 or later")
        print("- The model uses the Neural Engine when available on Apple Silicon")
        print("- Input images should be RGB format")
        print("- Images are automatically resized to 384x384 pixels")
        print("- Pixel values are normalized to [0, 1]")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()
