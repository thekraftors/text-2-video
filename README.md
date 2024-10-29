
# Text-to-Video Project

This project converts text inputs into animated videos by leveraging machine learning models. It uses libraries like `diffusers`, `moviepy`, `torch`, and `gradio` for the interface.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Gradio Interface](#gradio-interface)
- [Directory Structure](#directory-structure)
- [Configuration Files](#configuration-files)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Requirements
- Python 3.8 or later
- Required libraries (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://your-repo-link.git
   cd text-2-video
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify `torch` can detect your device:
   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   ```

## Usage

Run the main application:

```bash
python3 app.py
```

The script loads model configurations and dependencies to convert text into video animations. 

## Gradio Interface

You can use Gradio to create a user-friendly web interface. Start the Gradio server by adding the following to `app.py`:

```python
import gradio as gr

def generate_video(text):
    # Your text-to-video function
    return video_path

iface = gr.Interface(fn=generate_video, inputs="text", outputs="video")
iface.launch()
```

Then run:

```bash
python3 app.py
```

This will launch a Gradio interface for generating videos from text inputs.

## Directory Structure

- `app.py`: Main script for text-to-video generation.
- `requirements.txt`: Lists all required libraries.
- `models/`: Contains model files (e.g., `.safetensors` files).
- `config.json`: Configuration settings for the video generation.

## Configuration Files

- `config.json`: Specifies model configuration.
- `generation_config.json`: Sets generation parameters for animation.

## Troubleshooting

- **ModuleNotFoundError**: Install missing modules with `pip install <module_name>`.
- **RuntimeError for MPS devices**: Set the device in `app.py` to CPU or CUDA if on a compatible NVIDIA GPU.

## License
MIT License
