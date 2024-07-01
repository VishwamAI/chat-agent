# VishwamAI Documentation

## Overview

VishwamAI is an advanced language model based on the Transformer architecture, designed for various natural language processing tasks. The model aims for 100% accuracy in the MMLU benchmark, focusing on mathematical reasoning and other benchmarks. It integrates features like image to text, PDF to text, summarization, video to text, and audio to text.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/vishwam_ai.git
   cd vishwam_ai
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the VishwamAI model, run:

```
python scripts/train.py
```

This script will load the configuration from `configs/default_config.yaml`, initialize the model and datasets, and start the training process.

### Generating Text

To generate text using a trained model, use:

```
python scripts/generate_text.py --prompt "Your prompt here" --max_length 100
```

### Evaluating the Model

To evaluate the model on a test dataset, run:

```
python scripts/evaluate.py --test_file path/to/test/file.txt
```

## Configuration

You can modify the model and training configuration by editing the `configs/default_config.yaml` file.

## Features

### Image to Text

The `image_to_text` function extracts text from image files using Tesseract OCR. It supports `.png`, `.jpg`, and `.jpeg` formats.

### PDF to Text

The `pdf_to_text` function extracts text from PDF files using PyMuPDF.

### Summarization

The `summarize_text` function summarizes text using a pre-trained summarization model from Hugging Face.

### Video to Text

The `video_to_text` function extracts text from video files using Tesseract OCR. It supports various video formats such as `.mp4`, `.avi`, and `.mkv`.

### Audio to Text

The `audio_to_text` function converts audio files to text using the `pydub` and `speech_recognition` libraries. It supports various audio formats such as `.wav`, `.mp3`, and `.flac`.

## Documentation

For more detailed information about the model architecture, training process, and API reference, please refer to the `docs/` directory. Specifically, see the `docs/FEATURES.md` file for detailed information on the features.

## Contributing

Contributions to VishwamAI are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
