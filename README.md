# TrOCR Small Handwritten - CoreML

This repository contains a CoreML conversion of Microsoft's [microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten) model for **Apple Silicon** devices. The model performs **optical character recognition (OCR)** on single text-line handwritten images.

## Model Description

This is a **CoreML conversion** of the original **TrOCR model** introduced in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) by Li et al. The original model is an **encoder-decoder Transformer architecture**, which has been converted to run optimally on Apple Silicon devices.

Original model: [microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten)

## Conversion Details

- **Source Model**: [microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten)
- **Target Format**: CoreML
- **Supported Devices**: Apple Silicon Macs (M1/M2/M3)
- **Input**: RGB images (single text line)
- **Output**: Text transcription
- **CoreML Tools Version**: 8.1

## Performance

The model has been optimized for **Apple Silicon Neural Engine**. Performance metrics:
- **Memory Usage**: ~1.2GB during inference
- **Inference Time**: 150-200ms per image on M1/M2
- **Supported macOS versions**: macOS 13.0 or later
- **Model Size**: ~240MB

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ajmcclary/trocr-small-handwritten-coreml.git
cd trocr-small-handwritten-coreml
```

2. Run the setup and conversion script:
```bash
./setup_and_convert.sh
```

This will:
- Create a Python virtual environment
- Install required dependencies
- Download the TrOCR model
- Convert it to CoreML format
- Save as `TrOCR-Handwritten.mlpackage`

## Usage

### Testing the Conversion

The repository includes a test script that downloads a sample handwritten text image and runs inference:

```bash
python test_conversion.py
```

Example output:
```
Prediction Results:
--------------------------------------------------
Detected Text: inclusive " Mr. Bonn commented icily. " Let us have a
--------------------------------------------------
```

### Integration in Swift

```swift
import CoreML

// Load the model
let config = MLModelConfiguration()
config.computeUnits = .all  // Use Neural Engine when available
let model = try TrOCRSmallHandwritten(configuration: config)

// Prepare input image (must be RGB format, will be resized to 384x384)
let imageConstraint = model.modelDescription.inputDescriptionsByName["pixel_values"]!.imageConstraint!
let imageOptions: [MLFeatureValue.ImageOption: Any] = [
    .cropAndScale: VNImageCropAndScaleOption.scaleFit.rawValue
]

guard let inputImage = try? MLFeatureValue(
    imageAt: imageURL,
    constraint: imageConstraint,
    options: imageOptions
) else {
    fatalError("Failed to create input image")
}

// Create input dictionary
let inputFeatures = try! MLDictionary(dictionary: [
    "pixel_values": inputImage
])

// Get prediction
guard let output = try? model.prediction(from: inputFeatures) else {
    fatalError("Failed to get prediction")
}

// Process output tokens
let tokenIds = output.featureValue(for: "var_5238")!.multiArrayValue!
// Decode tokens to text using your tokenizer
```

## Model Details

The converted model includes the following optimizations:
- Input: RGB images (automatically resized to 384x384 pixels)
- Pixel normalization: Values scaled to [0, 1]
- Maximum sequence length: 20 tokens
- Temperature scaling (0.3) for focused sampling
- Token-level repetition penalty
- Pattern-based repetition detection (3-token window)
- Neural Engine optimization for Apple Silicon

## Limitations

- Maximum text length of ~20 words
- May struggle with very complex handwriting
- Requires macOS 13 or later
- Best performance on Apple Silicon Macs using Neural Engine
- Single text line recognition only (not suitable for paragraphs)
- Input images should be pre-cropped to contain only the text line
- No support for rotated or severely skewed text

## Preprocessing Requirements

1. Input images must be:
   - RGB format
   - Single line of text
   - Reasonably horizontal alignment
   - Good contrast between text and background
   - Will be automatically resized to 384x384 pixels

2. For best results:
   - Crop images tightly around the text line
   - Ensure good lighting and contrast
   - Minimize background noise/patterns
   - Avoid severe rotation or skewing

## License

This model conversion is released under the **MIT license**, following the original model's licensing. See the LICENSE file for more details.

## Attribution

This is a CoreML conversion of the [microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten) model created by Microsoft. Please cite the original work when using this model:

```bibtex
@misc{li2021trocr,
    title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models},
    author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
    year={2021},
    eprint={2109.10282},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Support

For issues specific to the CoreML conversion, please open an issue in this repository. For issues related to the original model, please refer to the [original repository](https://github.com/microsoft/unilm/tree/master/trocr).
