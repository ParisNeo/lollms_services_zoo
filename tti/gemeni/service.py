# Title LollmsGemini
# Licence: Apache 2.0
# Author : Paris Neo

from pathlib import Path
import sys
from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
import time
import io
import sys
import requests
import os
import base64
import subprocess
import time
import json
import platform
from dataclasses import dataclass
from PIL import Image, PngImagePlugin
from enum import Enum
from typing import List, Dict, Any, Optional

from ascii_colors import ASCIIColors, trace_exception
from lollms.paths import LollmsPaths
from lollms.utilities import PackageManager, find_next_available_filename
from lollms.tti import LollmsTTI
import subprocess
import shutil
from tqdm import tqdm
import threading
from io import BytesIO
import os
import pipmaster as pm

# Ensure google-generativeai is installed
if not pm.is_installed("google-genai"):
    pm.install("google-genai")
if not pm.is_installed("Pillow"):
    pm.install("Pillow")
    
try:
    from google import genai
    from google.genai import types
    from PIL import Image
except ImportError as e:
    ASCIIColors.error("Couldn't import google.generativeai. Please ensure it is installed.")
    trace_exception(e)
    # Optionally re-raise or exit if the import is critical for the module to load
    # raise e 


class LollmsGemini(LollmsTTI):
    def __init__(
        self, 
        app: LollmsApplication, 
        output_folder: Optional[str | Path] = None,
        config: Optional[dict] = None, # Added config parameter
        ):
        """
        Initializes the LollmsGemini binding.

        Args:
            app (LollmsApplication): The main Lollms application instance.
            output_folder (Path|str, optional): The output folder where generated images will be saved. Defaults to None.
            config (dict, optional): A configuration dictionary, usually loaded from a yaml file.
        """
        # Try to get the API key from environment variable first
        api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Define the configuration template
        config_template = ConfigTemplate([
            {"name": "api_key", "type": "str", "value": api_key, "help": "Your Google AI Studio API key."},
            {"name": "model_name", "type": "str", "value": "gemini-1.5-flash-latest", "options":["gemini-2.0-flash-exp-image-generation"], "help": "The specific Gemini model to use for image generation. Note: Check Google's documentation for models supporting image generation."},
            # Gemini API (as per provided snippet) doesn't explicitly take width/height, seed, steps etc.
            # These might be implicitly controlled or not available via this specific API endpoint/model.
            # Adding them as placeholders might be confusing, so omitting for now.
            {"name": "max_output_tokens", "type": "int", "value": 8192, "help":"Maximum number of tokens to generate"},

            # Other potential Gemini parameters could be added here if supported by the API
            # e.g., safety_settings, generation_config TBD
        ])
        
        # Initialize the BaseConfig part
        # If a config dict is provided, load it; otherwise, use defaults from the template
        base_config = BaseConfig(config=config or {"api_key": api_key, "model_name": "gemini-2.0-flash-exp-image-generation", "max_output_tokens":8192})

        # Create the TypedConfig
        service_config = TypedConfig(config_template, base_config)

        super().__init__("gemini", app, service_config, output_folder)
        self.client = None # Initialize client later in paint
        self.settings_updated()

    def settings_updated(self):
        """
        Called when the configuration settings are updated.
        Re-initializes the Gemini client with the new API key if it changed.
        """
        if self.client is None or (hasattr(genai, '_client') and genai._client is not None and self.service_config.api_key != genai.api_key):
             # Configure the client only if the key is provided
             if self.service_config.api_key:
                 try:
                     self.client = genai.Client(api_key=self.service_config.api_key)
                     # Create a client instance to potentially check authentication, though configure is usually sufficient
                     # self.client = genai.GenerativeModel(self.service_config.model_name) # Let's create the model in paint
                     ASCIIColors.info("Gemini API configured successfully.")
                 except Exception as e:
                     ASCIIColors.error(f"Failed to configure Gemini API: {e}")
                     trace_exception(e)
                     self.client = None # Ensure client is None if config fails
             else:
                 ASCIIColors.warning("Gemini API key is missing. Please configure it in the binding settings.")
                 self.client = None


    def paint(
        self,
        positive_prompt: str,
        negative_prompt: str, # Gemini might not directly support negative prompts in this API call
        sampler_name: str = "Euler", # Parameter unused by Gemini API in this example
        seed: Optional[int] = None, # Parameter unused by Gemini API in this example
        scale: Optional[float] = None, # Parameter unused by Gemini API in this example
        steps: Optional[int] = None, # Parameter unused by Gemini API in this example
        width: Optional[int] = None, # Parameter unused by Gemini API in this example (might be implicit)
        height: Optional[int] = None, # Parameter unused by Gemini API in this example (might be implicit)
        output_folder: Optional[str | Path] = None,
        output_file_name: Optional[str] = None,
        progress_callback: Optional[callable]=None, # Added progress callback
        **kwargs # Added to capture any unexpected arguments
    ) -> tuple[str | None, dict]:
        """
        Generates an image based on the positive prompt using the Gemini API.

        Args:
            positive_prompt (str): The text prompt to generate the image from.
            negative_prompt (str): The negative prompt (likely ignored by this Gemini endpoint).
            sampler_name (str, optional): Sampler name (ignored). Defaults to "Euler".
            seed (Optional[int], optional): Seed (ignored). Defaults to None.
            scale (Optional[float], optional): Scale (ignored). Defaults to None.
            steps (Optional[int], optional): Steps (ignored). Defaults to None.
            width (Optional[int], optional): Width (ignored). Defaults to None.
            height (Optional[int], optional): Height (ignored). Defaults to None.
            output_folder (Optional[str | Path], optional): Folder to save the output. Defaults to self.output_folder.
            output_file_name (Optional[str], optional): Specific filename for the output. Defaults to None.
            progress_callback (Optional[callable]): Callback function for progress updates.

        Returns:
            tuple[str | None, dict]: A tuple containing the path to the saved image file (or None if failed)
                                     and a dictionary containing metadata (like the prompt).
        """
        if not self.service_config.api_key:
            ASCIIColors.error("Gemini API key is not set. Please configure it in the binding settings.")
            return None, {"error": "API key not configured"}

        # Ensure client is configured
        self.settings_updated() # Call this to make sure API key is configured if it wasn't before

        output_folder = Path(output_folder or self.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Define the output file path
        if output_file_name:
            # Ensure it has a .png extension (or let PIL handle it)
            file_name = (output_folder / output_file_name).with_suffix(".png")
        else:
            file_name = find_next_available_filename(output_folder, "img_gemini_", extension="png")

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents="Generate an image from the following positive prompt:\n"+positive_prompt+"\nMake sure to avoid the following negative prompt:\n"+negative_prompt,
                config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
                )
            )
            full_text = ""
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(part.text)
                    full_text+=part.text
                elif part.inline_data is not None:
                    image = Image.open(BytesIO((part.inline_data.data)))

                    # Add metadata (like prompt) to the image using PngInfo
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("positive_prompt", positive_prompt)
                    if negative_prompt:
                        metadata.add_text("negative_prompt", negative_prompt)
                    if full_text:
                        metadata.add_text("gemini_text_response", full_text.strip())
                    metadata.add_text("model_name", self.service_config.model_name)
                    # Add other parameters if they were used/relevant
                    
                    image.save(file_name, pnginfo=metadata)
                    ASCIIColors.yellow(f"Image saved to {file_name}")
                    
                    if progress_callback:
                        progress_callback(100, "Image saved") # Final progress

                    return str(file_name), {
                        "positive_prompt": positive_prompt, 
                        "negative_prompt": negative_prompt,
                        "model_name": self.service_config.model_name,
                        "gemini_text_response": full_text.strip()
                        # Add other relevant metadata here
                        }
            else:
                error_message = f"No image data received from Gemini. Text response: {text_output.strip()}"
                ASCIIColors.red(error_message)
                if progress_callback:
                     progress_callback(-1, "No image data received") # Error progress
                return None, {"error": error_message, "gemini_text_response": text_output.strip()}

        except Exception as e:
            ASCIIColors.error(f"An error occurred during Gemini image generation: {e}")
            trace_exception(e)
            if progress_callback:
                progress_callback(-1, f"API Error: {e}") # Error progress
            # Try to parse more specific Google API errors if possible
            if hasattr(e, 'message'):
                error_msg = e.message
            else:
                error_msg = str(e)
            return None, {"error": f"Gemini API error: {error_msg}"}

    def paint_from_images(
        self, 
        positive_prompt: str, 
        images: List[str], 
        negative_prompt: str = "", 
        output_folder: Optional[str | Path] = None, # Added output folder
        output_file_name: Optional[str] = None, # Added output file name
        # Parameters like width/height/steps might be needed if an image variation API exists
        width: Optional[int] = None, 
        height: Optional[int] = None,
        steps: Optional[int] = None,
        scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs # Added to capture any unexpected arguments
        ) -> tuple[str | None, dict]:
        """
        Generates image variations or performs image-to-image tasks using Gemini.
        NOTE: As of the provided snippet, Gemini's standard text/image generation
        doesn't explicitly show image input. This method assumes such a capability
        might exist or will be added. Currently raises NotImplementedError.

        Args:
            positive_prompt (str): The guiding prompt for the image variation/task.
            images (List[str]): List of paths to input images. Gemini might support one or more.
            negative_prompt (str, optional): Negative prompt (support uncertain). Defaults to "".
            output_folder (Optional[str | Path], optional): Folder to save the output. Defaults to self.output_folder.
            output_file_name (Optional[str], optional): Specific filename for the output. Defaults to None.
            width (Optional[int], optional): Target width (support uncertain). Defaults to None.
            height (Optional[int], optional): Target height (support uncertain). Defaults to None.
            steps (Optional[int], optional): Steps (support uncertain). Defaults to None.
            scale (Optional[float], optional): Scale (support uncertain). Defaults to None.
            seed (Optional[int], optional): Seed (support uncertain). Defaults to None.


        Returns:
            tuple[str | None, dict]: Path to the generated image and metadata, or None and error dict.
        """
        # TODO: Implement this if/when Gemini API supports image input for variations/editing
        # This would involve:
        # 1. Checking the specific Gemini model and API documentation for image input capabilities.
        # 2. Loading the input image(s) from the `images` list.
        # 3. Converting image(s) to the format required by the API (e.g., bytes, base64).
        # 4. Making the appropriate API call with the prompt and image data.
        # 5. Processing the response to get the output image data.
        # 6. Saving the output image and returning the path and metadata.

        ASCIIColors.warning("paint_from_images is not implemented for the Gemini binding yet.")
        # raise NotImplementedError("Gemini image-to-image generation is not implemented in this binding.")
        return None, {"error": "paint_from_images is not implemented for the Gemini binding yet."}


    @staticmethod
    def get(app: LollmsApplication) -> 'LollmsGemini':
        """
        Static method to get an instance of the LollmsGemini class.

        Args:
            app (LollmsApplication): The main Lollms application instance.

        Returns:
            LollmsGemini: An instance of the LollmsGemini class.
        """
        # This allows the Lollms infrastructure to discover and instantiate the binding.
        # The __init__ method will handle the actual setup.
        # Passing None for output_folder and config initially; LollmsApp will manage setting these later
        # through configuration loading and potential user settings updates.
        return LollmsGemini(app=app, output_folder=None, config=None)