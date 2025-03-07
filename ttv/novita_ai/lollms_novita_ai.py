from pathlib import Path
from typing import List, Optional, Dict, Any
from lollms.ttv import LollmsTTV
from lollms.app import LollmsApplication
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
from lollms.utilities import find_next_available_filename
import requests
import json
import os
import time

class LollmsNovitaAITextToVideo(LollmsTTV):
    """
    A binding for the Novita.ai Text-to-Video API.
    This class allows generating videos from text prompts using the Novita.ai service.
    """
    def __init__(
                    self,
                    app:LollmsApplication,
                    output_folder:str|Path=None
                ):
        """
        Initializes the NovitaAITextToVideo binding.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the Novita.ai API. Defaults to "https://api.novita.ai/v3/async".
        """
        # Check for the NOVITA_AI_KEY environment variable if no API key is provided
        api_key = os.getenv("NOVITA_AI_KEY","")
        service_config = TypedConfig(
            ConfigTemplate([
                {"name":"api_key", "type":"str", "value":api_key, "help":"A valid Novita AI key to generate text using anthropic api"},
                {"name":"generation_engine","type":"str","value":"stable_diffusion", "options": ["stable_diffusion", "hunyuan-video-fast", "wan-t2v"], "help":"The engine name"},
                {"name":"sd_model_name","type":"str","value":"darkSushiMixMix_225D_64380.safetensors", "options": ["darkSushiMixMix_225D_64380.safetensors"], "help":"The model name"},
                {"name":"n_frames","type":"int","value":129, "help":"The number of frames in the video"},
                {"name":"guidance_scale", "type":"float", "value":7.5, "help":"The guidance scale for the generation"},
                {"name":"loras", "type":"str", "value":None, "help":"List of LoRA configurations"},
                {"name":"embeddings", "type":"str", "value":None, "help":"List of embedding configurations"},
                {"name":"closed_loop", "type":"bool", "value":False, "help":"Whether to use closed loop generation"},
                {"name":"clip_skip", "type":"int", "value":0, "help":"Number of layers to skip in CLIP"}
            ]),
            BaseConfig(config={
                "api_key": "",     # use avx2
            })
        )

        super().__init__("novita_ai", app, service_config,output_folder)
        self.sd_model_name = self.service_config.sd_model_name
        self.base_url = "https://api.novita.ai/v3/async"

        models = self.getModels()
        service_config.config_template["sd_model_name"]["options"] = [model["model_name"] for model in models]

    def settings_updated(self):
        models = self.getModels()
        self.service_config.config_template["sd_model_name"]["options"] = models

    def getModels(self):
        """
        Gets the list of models
        """
        url = "https://api.novita.ai/v3/model"
        headers = {
            "Content-Type": "<content-type>",
            "Authorization": f"Bearer {self.service_config.api_key}"
        }

        response = requests.request("GET", url, headers=headers)
        js = response.json()
        return js["models"]
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_name: str = "",
        height: int = 512,
        width: int = 512,
        steps: int = 20,
        seed: int = -1,
        nb_frames: int = None,
        output_dir:str | Path =None,
    ) -> str:
        """
        Generates a video from text prompts using the Novita.ai API.

        Args:
            model_name (str): Name of the model checkpoint.
            height (int): Height of the video, range [256, 1024].
            width (int): Width of the video, range [256, 1024].
            steps (int): Number of denoising steps, range [1, 50].
            prompts (List[Dict[str, Any]]): List of prompts with frames and text descriptions.
            negative_prompt (Optional[str]): Text input to avoid in the video. Defaults to None.
            seed (int): Random seed for reproducibility. Defaults to -1.
            guidance_scale (Optional[float]): Controls adherence to the prompt. Defaults to None.
            loras (Optional[List[Dict[str, Any]]]): List of LoRA parameters. Defaults to None.
            embeddings (Optional[List[Dict[str, Any]]]): List of embeddings. Defaults to None.
            closed_loop (Optional[bool]): Controls animation loop behavior. Defaults to None.
            clip_skip (Optional[int]): Number of layers to skip during optimization. Defaults to None.

        Returns:
            str: The task_id for retrieving the generated video.
        """
        if output_dir is None:
            output_dir = self.output_folder

        if nb_frames is None:
            nb_frames =self.service_config.n_frames

        if self.service_config.generation_engine=="hunyuan-video-fast":
            width, height, nb_frames, steps = self.pin_dimensions_frames_steps(width, height, nb_frames, steps)
            url = "https://api.novita.ai/v3/async/hunyuan-video-fast"

            payload = {
                "model_name": "hunyuan-video-fast",
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "prompt": prompt,
                "frames": nb_frames
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.service_config.api_key}"
            }

            response = requests.request("POST", url, json=payload, headers=headers)
        elif self.service_config.generation_engine=="wan-t2v":
            width, height, nb_frames, steps = self.pin_dimensions_frames_steps(width, height, nb_frames, steps)
            url = "https://api.novita.ai/v3/async/wan-t2v"

            payload = {
                "model_name": "wan-t2v",
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "prompt": prompt,
                "frames": nb_frames
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.service_config.api_key}"
            }

            response = requests.request("POST", url, json=payload, headers=headers)
            
        elif self.service_config.generation_engine=="stable_diffusion":
            print(response.text)
            if model_name=="":
                model_name = self.sd_model_name


            url = f"{self.base_url}/txt2video"
            headers = {
                "Authorization": f"Bearer {self.service_config.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "extra": {
                    "response_video_type": "mp4", # gif
                    "enterprise_plan": {"enabled": False}
                },
                "sd_model_name": model_name,
                "height": height,
                "width": width,
                "steps": steps,
                "prompts": [
                    {
                        "frames": nb_frames,
                        "prompt": prompt
                    }
                ],
                "negative_prompt": negative_prompt,
                "guidance_scale": self.service_config.guidance_scale,
                "seed": seed,
                "loras": self.service_config.loras,
                "embeddings": self.service_config.embeddings,
                "closed_loop": self.service_config.closed_loop,
                "clip_skip": self.service_config.clip_skip
            }  
            # Remove None values from the payload to avoid sending null fields
            payload = {k: v for k, v in payload.items() if v is not None}

            response = requests.post(url, headers=headers, data=json.dumps(payload))
        else:
            return "Unsupported engine name"
        
        response.raise_for_status()  # Raise an exception for HTTP errors
        task_id = response.json().get("task_id")


        url = f"https://api.novita.ai/v3/async/task-result?task_id={task_id}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.service_config.api_key}",
        }
        done = False
        while not done:
            response = requests.request("GET", url, headers=headers)
            infos = response.json()
            if infos["task"]["status"]=="TASK_STATUS_SUCCEED" or infos["task"]["status"]=="TASK_STATUS_FAILED":
                done = True
            time.sleep(1)
        if infos["task"]["status"]=="TASK_STATUS_SUCCEED":
            if output_dir:
                output_dir = Path(output_dir)
                file_name = output_dir/find_next_available_filename(output_dir, "vid_novita_","mp4")  # You can change the filename if needed
                self.download_video(infos["videos"][0]["video_url"], file_name )
                return file_name
        return None
    
    def pin_dimensions_frames_steps(self, width, height, nframes, steps):
        # Supported widths
        standard_widths = [480, 640, 720, 864, 1280]
        
        # Width-to-height mapping
        width_height_map = {
            480: [640, 864],    # 480 width supports 640 or 864 height
            640: [480],         # 640 width supports 480 height
            720: [1280],        # 720 width supports 1280 height
            864: [480],         # 864 width supports 480 height
            1280: [720]         # 1280 width supports 720 height
        }
        
        # Supported nframes
        standard_nframes = [85, 129]
        
        # Supported steps range
        min_steps, max_steps = 2, 30
        
        # Pin the width to the nearest standard width
        pinned_width = min(standard_widths, key=lambda x: abs(x - width))
        
        # Pin the height to the nearest supported height for the pinned width
        supported_heights = width_height_map[pinned_width]
        pinned_height = min(supported_heights, key=lambda x: abs(x - height))
        
        # Pin the nframes to the nearest standard nframes
        pinned_nframes = min(standard_nframes, key=lambda x: abs(x - nframes))
        
        # Pin the steps to the valid range (2 to 30)
        pinned_steps = max(min_steps, min(max_steps, steps))
        
        return pinned_width, pinned_height, pinned_nframes, pinned_steps
    def generate_video_by_frames(self, prompts: List[str], frames: List[int], negative_prompt: str, fps: int = 8, 
                        sd_model_name: str = "",
                        height: int = 512,
                        width: int = 512,
                        steps: int = 20,
                        seed: int = -1,
                        output_dir:str | Path =None,
                       ) -> str:
        """
        Generates a video from a list of prompts and corresponding frames.

        Args:
            prompts (List[str]): List of text prompts for each frame.
            frames (List[int]): List of frame indices corresponding to each prompt.
            negative_prompt (str): Text describing elements to avoid in the video.
            fps (int): Frames per second. Default is 8.
            num_inference_steps (int): Number of steps for the model to infer. Default is 50.
            guidance_scale (float): Controls how closely the model adheres to the prompt. Default is 6.0.
            seed (Optional[int]): Random seed for reproducibility. Default is None.

        Returns:
            str: The path to the generated video.
        """
        if sd_model_name=="":
            sd_model_name = self.sd_model_name
        if output_dir is None:
            output_dir = self.output_folder


        url = f"{self.base_url}/txt2video"
        headers = {
            "Authorization": f"Bearer {self.service_config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "extra": {
                "response_video_type": "mp4", # gif
                "enterprise_plan": {"enabled": False}
            },
            "sd_model_name": sd_model_name,
            "height": height,
            "width": width,
            "steps": steps,
            "prompts": [
                {
                    "frames": nb_frames,
                    "prompt": prompt,
                    
                } for nb_frames, prompt in zip(prompts, frames)
            ],
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "loras": [],
            "embeddings": [],
            "closed_loop": closed_loop,
            "clip_skip": clip_skip
        }  
        # Remove None values from the payload to avoid sending null fields
        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors
        task_id = response.json().get("task_id")


        url = f"https://api.novita.ai/v3/async/task-result?task_id={task_id}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.service_config.api_key}",
        }
        done = False
        while not done:
            response = requests.request("GET", url, headers=headers)
            infos = response.json()
            if infos["task"]["status"]=="TASK_STATUS_SUCCEED" or infos["task"]["status"]=="TASK_STATUS_FAILED":
                done = True
            time.sleep(1)
        if infos["task"]["status"]=="TASK_STATUS_SUCCEED":
            if output_dir:
                output_dir = Path(output_dir)
                file_name = output_dir/find_next_available_filename(output_dir, "vid_novita_","mp4")  # You can change the filename if needed
                self.download_video(infos["videos"][0]["video_url"], file_name )
                return file_name
        return None

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Retrieves the result of a video generation task using the task_id.

        Args:
            task_id (str): The task_id returned by the generate_video method.

        Returns:
            Dict[str, Any]: The task result containing the video URL and other details.
        """
        url = f"{self.base_url}/task-result"
        headers = {
            "Authorization": f"Bearer {self.service_config.api_key}",
        }
        params = {
            "task_id": task_id,
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        return response.json()

    def download_video(self, video_url: str, save_path: Path) -> None:
        """
        Downloads the generated video from the provided URL and saves it to the specified path.

        Args:
            video_url (str): The URL of the video to download.
            save_path (Path): The path where the video will be saved.
        """
        response = requests.get(video_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(save_path, "wb") as file:
            file.write(response.content)
