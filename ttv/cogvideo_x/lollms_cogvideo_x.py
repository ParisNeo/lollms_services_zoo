import os
import time
from pathlib import Path
from typing import List, Optional
import pipmaster as pm

# Install required libraries if not already present
if not pm.is_installed("torch"):
    pm.install_multiple(["torch","torchvision"," torchaudio"], "https://download.pytorch.org/whl/cu118")  # Adjust CUDA version as needed
if not pm.is_installed("diffusers"):
    pm.install("diffusers")
if not pm.is_installed("transformers"):
    pm.install("transformers")
if not pm.is_installed("accelerate"):
    pm.install("accelerate")
if not pm.is_installed("imageio-ffmpeg"):
    pm.install("imageio-ffmpeg")

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from lollms.app import LollmsApplication
from lollms.main_config import LOLLMSConfig
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
from lollms.utilities import find_next_available_filename
from lollms.ttv import LollmsTTV
from ascii_colors import ASCIIColors

class LollmsCogVideoX(LollmsTTV):
    """
    LollmsCogVideoX is an implementation of LollmsTTV using CogVideoX for text-to-video generation.
    """
    
    def __init__(
            self,
            app: LollmsApplication,
            output_folder: str | Path = None
    ):
        # Define service configuration
        service_config = TypedConfig(
            ConfigTemplate([
                {"name": "model_name", "type": "str", "value": "THUDM/CogVideoX-2b", "options": ["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"], "help": "CogVideoX model to use"},
                {"name": "use_gpu", "type": "bool", "value": True, "help": "Use GPU if available"},
                {"name": "dtype", "type": "str", "value": "float16", "options": ["float16", "bfloat16"], "help": "Data type for model precision"},
            ]),
            BaseConfig(config={
                "model_name": "THUDM/CogVideoX-2b",  # Default to 2B model (less VRAM-intensive)
                "use_gpu": True,
                "dtype": "float16",
            })
        )
        super().__init__("cogvideox", app, service_config, output_folder)

        # Initialize CogVideoX pipeline
        self.pipeline = None
        self.load_pipeline()

    def load_pipeline(self):
        """Loads or reloads the CogVideoX pipeline based on config."""
        try:
            dtype = torch.float16 if self.service_config.dtype == "float16" else torch.bfloat16
            self.pipeline = CogVideoXPipeline.from_pretrained(
                self.service_config.model_name,
                torch_dtype=dtype
            )
            if self.service_config.use_gpu and torch.cuda.is_available():
                self.pipeline.to("cuda")
                self.pipeline.enable_model_cpu_offload()  # Optimize VRAM usage
            else:
                ASCIIColors.warning("GPU not available or disabled. Running on CPU (slower).")
            ASCIIColors.success(f"Loaded CogVideoX model: {self.service_config.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CogVideoX pipeline: {str(e)}")

    def settings_updated(self):
        """Reloads the pipeline if settings change."""
        self.load_pipeline()

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_name: str = "",
        height: int = 480,
        width: int = 720,
        steps: int = 50,
        seed: int = -1,
        nb_frames: int = 49,
        output_dir: str | Path = None,
    ) -> str:
        """
        Generates a video from a text prompt using CogVideoX.

        Args:
            prompt (str): The text prompt describing the video content.
            negative_prompt (Optional[str]): Ignored (CogVideoX doesn't support it natively).
            model_name (str): Overrides config model if provided (optional).
            height (int): Desired height of the video (default 480).
            width (int): Desired width of the video (default 720).
            steps (int): Number of inference steps (default 50).
            seed (int): Random seed (default -1 for random).
            nb_frames (int): Number of frames (default 49, ~6 seconds at 8 fps).
            output_dir (str | Path): Optional custom output directory.

        Returns:
            str: Path to the generated video file.
        """
        output_path = Path(output_dir) if output_dir else self.output_folder
        output_path.mkdir(exist_ok=True, parents=True)

        # Handle unsupported parameters
        if negative_prompt:
            ASCIIColors.warning("Warning: CogVideoX does not support negative prompts. Ignoring negative_prompt.")
        if model_name and model_name != self.service_config.model_name:
            ASCIIColors.warning(f"Overriding config model {self.service_config.model_name} with {model_name}")
            self.service_config.model_name = model_name
            self.load_pipeline()

        # Generation parameters
        gen_params = {
            "prompt": prompt,
            "num_frames": nb_frames,
            "num_inference_steps": steps,
            "guidance_scale": 6.0,  # Default value from CogVideoX docs
            "height": height,
            "width": width,
        }
        if seed != -1:
            gen_params["generator"] = torch.Generator(device="cuda" if self.service_config.use_gpu else "cpu").manual_seed(seed)

        # Generate video
        try:
            ASCIIColors.info("Generating video with CogVideoX...")
            start_time = time.time()
            video_frames = self.pipeline(**gen_params).frames[0]  # CogVideoX returns a list of frame batches
            output_filename = find_next_available_filename(output_path, "cogvideox_output.mp4")
            export_to_video(video_frames, output_filename, fps=8)
            elapsed_time = time.time() - start_time
            ASCIIColors.success(f"Video generated and saved to {output_filename} in {elapsed_time:.2f} seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to generate video: {str(e)}")

        return str(output_filename)

    def generate_video_by_frames(self, prompts: List[str], frames: List[int], negative_prompt: str, fps: int = 8, 
                                num_inference_steps: int = 50, guidance_scale: float = 6.0, 
                                seed: Optional[int] = None) -> str:
        """
        Generates a video from a list of prompts. Since CogVideoX doesn't natively support multi-prompt videos,
        this concatenates prompts into a single description.

        Args:
            prompts (List[str]): List of prompts for each segment.
            frames (List[int]): Number of frames per segment (summed to total frames).
            negative_prompt (str): Ignored.
            fps (int): Frames per second (default 8).
            num_inference_steps (int): Inference steps (default 50).
            guidance_scale (float): Guidance scale (default 6.0).
            seed (Optional[int]): Random seed.

        Returns:
            str: Path to the generated video file.
        """
        if not prompts or not frames:
            raise ValueError("Prompts and frames lists cannot be empty.")
        
        # Combine prompts into a single narrative
        combined_prompt = " ".join(prompts)
        total_frames = sum(frames)
        
        return self.generate_video(
            prompt=combined_prompt,
            negative_prompt=negative_prompt,
            steps=num_inference_steps,
            seed=seed if seed is not None else -1,
            nb_frames=total_frames
        )

    def getModels(self):
        """Returns available CogVideoX models."""
        return ["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"]