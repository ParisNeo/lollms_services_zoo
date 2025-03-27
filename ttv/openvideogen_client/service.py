import requests
import time
from typing import List, Optional
from pathlib import Path
from lollms.app import LollmsApplication
from lollms.main_config import LOLLMSConfig
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
from lollms.utilities import find_next_available_filename
from lollms.ttv import LollmsTTV
from ascii_colors import ASCIIColors

class LollmsOpenVideoGenClient(LollmsTTV):
    """
    LollmsOpenVideoGen is an implementation of LollmsTTV that interacts with the OpenVideoGen FastAPI server
    for text-to-video generation using job-based asynchronous endpoints.
    """
    
    def __init__(
            self,
            app: LollmsApplication,
            output_folder: str | Path = None
    ):
        # Define service configuration
        service_config = TypedConfig(
            ConfigTemplate([
                {"name": "api_url", "type": "str", "value": "http://localhost:8088", "help": "URL of the OpenVideoGen FastAPI server"},
                {"name": "model_name", "type": "str", "value": "cogvideox_2b", "options": [], "help": "Model to use for video generation (fetched dynamically)"},
                {"name": "timeout", "type": "int", "value": 600, "help": "Timeout for API requests in seconds (increased for video generation)"},
                {"name": "poll_interval", "type": "int", "value": 5, "help": "Interval in seconds to poll job status"},
            ]),
            BaseConfig(config={
                "api_url": "http://localhost:8088",
                "model_name": "cogvideox_2b",
                "timeout": 600,
                "poll_interval": 5,
            })
        )
        super().__init__("openvideogen", app, service_config, output_folder)

        # Fetch available models from the server
        self.available_models = self.get_available_models()
        if self.available_models:
            self.service_config.config_template["model_name"]["options"] = self.available_models
            if self.service_config.model_name not in self.available_models:
                self.service_config.model_name = self.available_models[0]
                ASCIIColors.warning(f"Model {self.service_config.model_name} not found. Using {self.available_models[0]} instead.")
        else:
            ASCIIColors.error("No models available from the OpenVideoGen server. Please ensure the server is running.")
            self.available_models = ["cogvideox_2b"]  # Fallback

    def get_available_models(self) -> List[str]:
        """Fetches the list of available models from the OpenVideoGen server."""
        try:
            response = requests.get(f"{self.service_config.api_url}/models", timeout=10)  # Short timeout for model fetch
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            ASCIIColors.error(f"Failed to fetch models from OpenVideoGen server: {str(e)}")
            return []

    def settings_updated(self):
        """Called when settings are updated. Refetch models if needed."""
        self.available_models = self.get_available_models()
        if self.available_models:
            self.service_config._template[1]["options"] = self.available_models
            if self.service_config.model_name not in self.available_models:
                self.service_config.model_name = self.available_models[0]
                ASCIIColors.warning(f"Model {self.service_config.model_name} not found. Using {self.available_models[0]} instead.")

    def _wait_for_job_completion(self, job_id: str, output_folder:Path, output_file_name:str) -> str:
        """Polls the job status until completion and downloads the video."""
        output_path = self.output_folder
        output_path.mkdir(exist_ok=True, parents=True)

        while True:
            try:
                status_response = requests.get(
                    f"{self.service_config.api_url}/status/{job_id}",
                    timeout=self.service_config.timeout
                )
                status_response.raise_for_status()
                status = status_response.json()

                job_status = status.get("status")
                progress = status.get("progress", 0)
                message = status.get("message", "")

                ASCIIColors.info(f"Job {job_id} status: {job_status}, Progress: {progress}% - {message}")

                if job_status == "completed":
                    # Download the video
                    download_url = f"{self.service_config.api_url}/download/{job_id}"
                    video_response = requests.get(download_url, timeout=self.service_config.timeout)
                    video_response.raise_for_status()
                    if output_file_name:
                        output_filename = output_path/output_file_name
                    else:
                        output_filename = find_next_available_filename(output_path, f"video_{job_id}.mp4")
                    with open(output_filename, "wb") as f:
                        f.write(video_response.content)
                    ASCIIColors.success(f"Video downloaded to {output_filename}")
                    return str(output_filename)

                elif job_status == "failed":
                    raise RuntimeError(f"Job failed: {message}")

                time.sleep(self.service_config.poll_interval)  # Poll interval

            except Exception as e:
                ASCIIColors.error(f"Error while polling job {job_id}: {str(e)}")
                raise

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
        fps: int = 8,
        output_folder: str | Path = None,
        output_file_name: str = None,
    ) -> str:
        """
        Submits a video generation job to the OpenVideoGen server and waits for completion.

        Args:
            prompt (str): The text prompt describing the video content.
            negative_prompt (Optional[str]): Negative prompt (if supported by the model).
            model_name (str): Overrides config model if provided (optional).
            height (int): Desired height of the video (default 480).
            width (int): Desired width of the video (default 720).
            steps (int): Number of inference steps (default 50).
            seed (int): Random seed (default -1 for random).
            nb_frames (int): Number of frames (default 49, ~6 seconds at 8 fps).
            fps (int): Frames per second (default 8).
            output_folder (str | Path): Optional custom output directory.
            output_file_name (str): Optional custom output file name (not used in this implementation).

        Returns:
            str: Path to the generated video file.
        """
        output_folder = Path(output_folder) if output_folder else self.output_folder
        selected_model = model_name if model_name else self.service_config.model_name

        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_name": selected_model,
            "height": height,
            "width": width,
            "steps": steps,
            "seed": seed,
            "nb_frames": nb_frames,
            "fps": fps,
        }

        # Submit the job
        try:
            ASCIIColors.info(f"Submitting video generation job with model {selected_model}...")
            response = requests.post(
                f"{self.service_config.api_url}/submit",
                json=payload,
                timeout=self.service_config.timeout
            )
            response.raise_for_status()
            result = response.json()
            job_id = result.get("job_id")
            if not job_id:
                raise ValueError("No job ID returned from the server.")
            ASCIIColors.success(f"Job submitted successfully: {job_id}")

            # Wait for completion and download the video
            return self._wait_for_job_completion(job_id, output_folder, output_file_name)

        except Exception as e:
            ASCIIColors.error(f"Failed to generate video: {str(e)}")
            raise RuntimeError(f"Failed to generate video: {str(e)}")

    def generate_video_by_frames(
        self,
        prompts: List[str],
        frames: List[int],
        negative_prompt: str = None,
        fps: int = 8,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None
    ) -> str:
        """
        Submits a multi-prompt video generation job to the OpenVideoGen server and waits for completion.

        Args:
            prompts (List[str]): List of prompts for each segment.
            frames (List[int]): Number of frames per segment (summed to total frames).
            negative_prompt (str): Negative prompt (if supported).
            fps (int): Frames per second (default 8).
            num_inference_steps (int): Inference steps (default 50).
            guidance_scale (float): Guidance scale (default 6.0).
            seed (Optional[int]): Random seed.

        Returns:
            str: Path to the generated video file.
        """
        if not prompts or not frames or len(prompts) != len(frames):
            raise ValueError("Prompts and frames lists must be non-empty and of equal length.")

        # Prepare the request payload
        payload = {
            "prompts": prompts,
            "frames": frames,
            "negative_prompt": negative_prompt,
            "fps": fps,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed if seed is not None else -1,
            "model_name": self.service_config.model_name,
        }

        # Submit the multi-prompt job
        try:
            ASCIIColors.info(f"Submitting multi-prompt video generation job with model {self.service_config.model_name}...")
            response = requests.post(
                f"{self.service_config.api_url}/submit_multi",
                json=payload,
                timeout=self.service_config.timeout
            )
            response.raise_for_status()
            result = response.json()
            job_id = result.get("job_id")
            if not job_id:
                raise ValueError("No job ID returned from the server.")
            ASCIIColors.success(f"Multi-prompt job submitted successfully: {job_id}")

            # Wait for completion and download the video
            return self._wait_for_job_completion(job_id)

        except Exception as e:
            ASCIIColors.error(f"Failed to generate multi-prompt video: {str(e)}")
            raise RuntimeError(f"Failed to generate multi-prompt video: {str(e)}")

    def getModels(self) -> List[str]:
        """Returns the list of available models from the OpenVideoGen server."""
        return self.available_models