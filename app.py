import os
import base64
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from moviepy.editor import VideoFileClip
import gradio as gr

# Device and data type configuration
device = "mps"
dtype = torch.float16

# Animation generation settings
step = 8  # Options: [1, 2, 4, 8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base_model = "emilianJR/epiCRealism"  # Select your preferred base model.

# Initialize motion adapter and pipeline
adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base_model, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

# Create output directory for videos
output_dir = 'video'
os.makedirs(output_dir, exist_ok=True)

def generate_video(prompt: str) -> tuple:
    """
    Generates an animation video based on the provided prompt.

    Args:
        prompt (str): The description of the animation to generate.

    Returns:
        tuple: A message and the path to the generated video.
    """
    if not prompt:
        return "No query provided", None

    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)

    gif_path = os.path.join(output_dir, 'animation.gif')
    export_to_gif(output.frames[0], gif_path)

    mp4_path = os.path.join(output_dir, 'animation.mp4')
    gif_clip = VideoFileClip(gif_path)

    # Ensure FPS is set; default to 24 if not available
    fps = gif_clip.fps or 24  
    gif_clip.write_videofile(mp4_path, codec="libx264", fps=fps)

    return None, mp4_path

# Gradio Interface setup
with gr.Blocks() as demo:
    gr.Markdown("## Kraftors AI Video Generator")
    
    with gr.Row():
        text_input = gr.Textbox(label="Enter your prompt:", placeholder="Describe the animation you want to generate...")
        video_output = gr.Video(label="Generated Animation")

    generate_btn = gr.Button("Generate Video")

    def gradio_generate(prompt: str) -> tuple:
        """
        Handles the generation of the video for the Gradio interface.

        Args:
            prompt (str): The input prompt for animation generation.

        Returns:
            tuple: A message and the path to the generated video.
        """
        message, video_path = generate_video(prompt)
        return (message, None) if message else (None, video_path)

    generate_btn.click(gradio_generate, inputs=text_input, outputs=[gr.Label(), video_output])

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=5000)
