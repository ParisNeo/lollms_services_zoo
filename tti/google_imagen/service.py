# Title LollmsGoogle
# Licence: Apache 2.0
# Author : Paris Neo & Contributing Community
# This binding uses the Google Cloud Vertex AI API for image generation.
# Ensure you have authenticated with Google Cloud:
# 1. Install gcloud SDK: https://cloud.google.com/sdk/docs/install
# 2. Login: `gcloud auth application-default login` (RECOMMENDED for local use)
# Or set the GOOGLE_APPLICATION_CREDENTIALS environment variable to your service account key file path.

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
import importlib # Keep for checking google-cloud-aiplatform import after installation

from ascii_colors import ASCIIColors, trace_exception
from lollms.paths import LollmsPaths
from lollms.utilities import PackageManager, find_next_available_filename, short_desc
from lollms.tti import LollmsTTI
import subprocess
import shutil
from tqdm import tqdm
import threading
from io import BytesIO
import pipmaster as pm # Use pipmaster directly as requested

# Ensure the required Google Cloud package is installed using pipmaster
if not pm.is_installed("google-cloud-aiplatform"):
    ASCIIColors.info("google-cloud-aiplatform package not found. Installing...")
    pm.install("google-cloud-aiplatform", force_reinstall=False)
    # Attempt to import again after installation to be sure
    try:
        importlib.import_module("google.cloud.aiplatform")
        ASCIIColors.success("google-cloud-aiplatform installed and imported successfully.")
    except ImportError as e:
        ASCIIColors.error("Failed to import google-cloud-aiplatform even after attempting installation.")
        trace_exception(e)
        raise e from e # Raise error if import fails after install attempt


# Import necessary Google Cloud libraries after ensuring installation
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel, ImageGenerationResponse
    # Import auth exceptions for better handling later
    from google.auth import exceptions as google_auth_exceptions

except ImportError as e:
    ASCIIColors.error("google-cloud-aiplatform library should be installed, but import failed.")
    ASCIIColors.error("There might be a version conflict or incomplete installation.")
    trace_exception(e)
    # Raise or handle gracefully depending on Lollms requirements
    raise e from e


class LollmsGoogle(LollmsTTI):
    def __init__(
            self,
            app: LollmsApplication,
            output_folder: Optional[str | Path] = None,
            config: Optional[dict] = None, # Added for config loading
            ):
        """
        Initializes the LollmsGoogle binding.

        Args:
            app (LollmsApplication): The LollmsApplication instance.
            output_folder (Path|str, optional): The output folder for generated images. Defaults to None.
            config (dict, optional): A configuration dictionary for the service.
        """
        default_project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        default_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")


        # Now create TypedConfig, applying the template and defaults
        # Values from loaded _config will override template defaults where names match.
        service_config = TypedConfig(
            ConfigTemplate([
                {"name": "project_id", "type": "str", "value": default_project_id, "help": "Your Google Cloud Project ID. Vertex AI API must be enabled."},
                {"name": "location", "type": "str", "value": default_location, "help": "The Google Cloud region for Vertex AI (e.g., 'us-central1')."},
                {"name": "model_id", "type": "str", "value": "imagegeneration@005", "options":["imagegeneration@006","imagegeneration@005","imagegeneration@002"], "help": "The specific Imagen model ID."},
                {"name": "guidance_scale", "type": "int", "value": 7, "min":1, "max":30, "help": "Controls prompt adherence (higher means stricter)."},
                {"name": "number_of_images", "type": "int", "value": 1, "min":1, "max":8, "help": "Number of images per request (max 8 for Imagen 2)."},
                {"name": "aspect_ratio", "type": "str", "value":"1:1", "options":["1:1", "16:9", "9:16", "4:3", "3:4"], "help":"Image aspect ratio."},
                {"name":"seed", "type":"int", "value":-1, "help":"Seed for generation (-1 for random)."},
            ])
        )

        # Ensure project_id is present after loading from config and template
        # If still empty after TypedConfig init (meaning not in passed config AND env var was empty), warn user.
        if not service_config.project_id:
            # The template already tried loading from env var, so if it's empty now, it wasn't set.
            ASCIIColors.warning("Google Cloud Project ID is not set in config or GOOGLE_CLOUD_PROJECT environment variable.")
            # It might still work if set via gcloud config, but explicit config is better.

        super().__init__("google_imagen", app, service_config, output_folder) # Use specific name
        self.vertex_ai_initialized = False

    def _ensure_output_folder(self, output_folder: Optional[str | Path] = None) -> Path:
        """Ensures the output folder exists and returns its Path object."""
        if output_folder is None:
            output_folder_path = self.output_folder # Use instance default
        else:
            output_folder_path = Path(output_folder)

        if not output_folder_path:
             # Fallback if self.output_folder was also None/empty during init
             # Ensure self.app and self.app.lollms_paths are valid
             if hasattr(self.app, "lollms_paths") and self.app.lollms_paths:
                 output_folder_path = self.app.lollms_paths.personal_outputs_path / "tti" / "google_imagen"
                 ASCIIColors.warning(f"Output folder not specified, using default: {output_folder_path}")
                 self.output_folder = output_folder_path # Update instance default if needed
             else:
                 # Absolute fallback if paths aren't available (shouldn't happen in normal Lollms run)
                 output_folder_path = Path("./outputs/tti/google_imagen")
                 ASCIIColors.error("LollmsPaths not available, using relative fallback output path.")

        output_folder_path.mkdir(parents=True, exist_ok=True)
        return output_folder_path


    def _initialize_vertex_ai(self):
        """Initializes the Vertex AI client if not already done. Relies on external authentication."""
        if self.vertex_ai_initialized:
            return True

        project_id = self.service_config.project_id
        location = self.service_config.location

        if not project_id:
            self.app.error("Google Cloud Project ID is missing in configuration.")
            self.app.error("Please set 'project_id' in the service settings or the GOOGLE_CLOUD_PROJECT env var.")
            return False
        if not location:
            self.app.error("Google Cloud Location is missing in configuration.")
            self.app.error("Please set 'location' in the service settings or the GOOGLE_CLOUD_LOCATION env var.")
            return False

        try:
            ASCIIColors.info("Initializing Vertex AI SDK...")
            ASCIIColors.info(f"  Project: {project_id}")
            ASCIIColors.info(f"  Location: {location}")
            ASCIIColors.info("  Attempting to use Application Default Credentials (ADC)...")
            ASCIIColors.info("  Ensure you have run 'gcloud auth application-default login' OR set GOOGLE_APPLICATION_CREDENTIALS.")

            # vertexai.init() uses ADC or GOOGLE_APPLICATION_CREDENTIALS by default.
            vertexai.init(project=project_id, location=location)

            # Optional lightweight check (can be commented out if causing issues/costs)
            try:
                from google.cloud import aiplatform
                # Ensure the endpoint matches the location for the client
                client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
                client = aiplatform.gapic.ModelServiceClient(client_options=client_options)
                parent = f"projects/{project_id}/locations/{location}"
                # Make a minimal request, e.g., list models with a limit
                request = aiplatform.gapic.ListModelsRequest(parent=parent, page_size=1)
                client.list_models(request=request)
                ASCIIColors.info("Initial authentication check via list_models seems okay.")
            except google_auth_exceptions.DefaultCredentialsError as auth_err:
                 self.app.error("Authentication failed during initial SDK check (DefaultCredentialsError).")
                 self.app.error("Please verify ADC ('gcloud auth application-default login') or GOOGLE_APPLICATION_CREDENTIALS.")
                 trace_exception(auth_err)
                 return False
            except google_auth_exceptions.GoogleAuthError as auth_err:
                 self.app.error("Authentication failed during initial SDK check (GoogleAuthError).")
                 self.app.error("Verify credentials and permissions.")
                 trace_exception(auth_err)
                 return False
            except Exception as api_err:
                 # This could be API not enabled, invalid project/location, network issues etc.
                 self.app.error(f"Error during initial Vertex AI check (Project:{project_id}, Location:{location}).")
                 self.app.error("Verify Project ID, Location, Vertex AI API enablement, and network connectivity.")
                 trace_exception(api_err)
                 # Decide if this should block initialization or just warn
                 return False # Treat check failure as initialization failure


            self.vertex_ai_initialized = True
            ASCIIColors.success("Vertex AI SDK initialized (credentials will be fully verified on first API call).")
            return True
        except google_auth_exceptions.GoogleAuthError as e: # Catch auth errors specifically during vertexai.init() itself
            self.app.error(f"Vertex AI Initialization Failed: Authentication Error during vertexai.init().")
            self.app.error(f"Ensure you have run 'gcloud auth application-default login' OR set the GOOGLE_APPLICATION_CREDENTIALS environment variable.")
            trace_exception(e)
            self.vertex_ai_initialized = False
            return False
        except Exception as e:
            self.app.error(f"Failed to initialize Vertex AI SDK due to an unexpected error.")
            self.app.error(f"Verify:")
            self.app.error(f"  1. 'google-cloud-aiplatform' is installed correctly.")
            self.app.error(f"  2. Project ID ('{project_id}') and Location ('{location}') are correct.")
            self.app.error(f"  3. Vertex AI API is enabled in your Google Cloud project.")
            self.app.error(f"  4. Network connectivity to Google Cloud APIs.")
            trace_exception(e)
            self.vertex_ai_initialized = False
            return False

    def settings_updated(self):
        """Called when settings are updated. Re-initializes Vertex AI if project/location changed."""
        ASCIIColors.info("Settings updated. Vertex AI will re-initialize on the next request if project/location changed.")
        self.vertex_ai_initialized = False

    def paint(
        self,
        positive_prompt: str,
        negative_prompt: str,
        sampler_name: str = "Default", # Imagen doesn't expose sampler choice like SD
        seed: Optional[int] = None,
        scale: Optional[float] = None, # Maps to guidance_scale
        steps: Optional[int] = None,   # Imagen doesn't use 'steps' in the same way
        width: Optional[int] = None,   # Controlled by aspect_ratio config
        height: Optional[int] = None,  # Controlled by aspect_ratio config
        output_folder: Optional[str | Path] = None,
        output_file_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates image(s) using Google Vertex AI Imagen.
        Returns a list of dictionaries [{'path': 'path/to/img.png', 'metadata': {...}}, ...], or [] on failure.
        """
        if not self._initialize_vertex_ai():
            self.app.error("Vertex AI initialization failed or not complete. Cannot generate image.")
            return [] # Return empty list as required by the expected signature

        output_folder_path = self._ensure_output_folder(output_folder)

        model_id = self.service_config.model_id
        guidance_scale = int(scale) if scale is not None else self.service_config.guidance_scale
        num_images = self.service_config.number_of_images
        aspect_ratio = self.service_config.aspect_ratio
        _seed = int(seed) if seed is not None and seed != -1 else self.service_config.seed
        api_seed_param = _seed if _seed != -1 else None # API expects None for random

        # Clamp guidance scale if needed (config UI should ideally handle this range)
        guidance_scale = max(1, min(guidance_scale, 30)) # Adjust max based on model specs if necessary

        try:
            ASCIIColors.info(f"Loading Imagen model reference: {model_id}")
            model = ImageGenerationModel.from_pretrained(model_id) # Gets client reference
            ASCIIColors.info(f"Generating {num_images} image(s) with Google Imagen...")
            self.app.info(f"Generating image with Google Imagen...\nPrompt: {short_desc(positive_prompt)}")

            generation_params = {
                "prompt": positive_prompt.strip(),
                "number_of_images": num_images,
                "guidance_scale": guidance_scale,
                "aspect_ratio": aspect_ratio,
                **({"seed": api_seed_param} if api_seed_param is not None else {}),
                **({"negative_prompt": negative_prompt.strip()} if negative_prompt and negative_prompt.strip() else {}),
            }
            ASCIIColors.debug(f"Generation parameters: {generation_params}")

            start_time = time.time()
            # === THE ACTUAL API CALL WHERE AUTH IS CRITICAL ===
            response: ImageGenerationResponse = model.generate_images(**generation_params)
            # ====================================================
            elapsed_time = time.time() - start_time
            ASCIIColors.info(f"Image generation API call completed in {elapsed_time:.2f} seconds.")

            results = []
            if not hasattr(response, 'images') or not response.images:
                 # Check if response object is different or if images list is empty
                 self.app.warning("API call succeeded but returned no images or response structure unexpected.")
                 ASCIIColors.debug(f"Raw API response (or type): {type(response)}") # Log response type for debugging
                 return []

            for i, image in enumerate(response.images):
                # Determine filename
                if output_file_name:
                    base_name = Path(output_file_name).stem
                    ext = Path(output_file_name).suffix if Path(output_file_name).suffix else ".png"
                    current_output_file_name = f"{base_name}_{i}{ext}" if num_images > 1 else f"{base_name}{ext}"
                    file_path = output_folder_path / current_output_file_name
                else:
                    file_path = find_next_available_filename(output_folder_path, "google_img_", extension="png")

                try:
                    # Accessing _image_bytes uses a private attribute - may break in future SDKs.
                    # Check if a public method exists in the `image` object's documentation.
                    if hasattr(image, '_image_bytes') and image._image_bytes:
                        image_bytes = image._image_bytes
                        with open(file_path, "wb") as f:
                            f.write(image_bytes)
                        ASCIIColors.yellow(f"Image {i+1}/{num_images} saved to: {file_path}")

                        # Prepare metadata
                        metadata = {
                            "positive_prompt": positive_prompt,
                            "negative_prompt": negative_prompt,
                            "model": model_id,
                            "guidance_scale": guidance_scale,
                            "aspect_ratio": aspect_ratio,
                            "seed": _seed, # Report the seed used (-1 means random was chosen by API)
                            "generation_time_sec": round(elapsed_time / num_images, 2) if num_images > 0 else elapsed_time
                        }
                        if hasattr(image, 'safety_attributes'):
                            metadata["safety_attributes"] = getattr(image, 'safety_attributes', None)

                        results.append({"path": str(file_path), "metadata": metadata})
                    else:
                         self.app.error(f"Could not access image bytes for image {i+1}. Image object structure might have changed or data is missing.")
                         ASCIIColors.warning(f"Image object type: {type(image)}, Attributes: {dir(image)}")

                except AttributeError as ae:
                    self.app.error(f"Attribute error accessing image data for image {i+1}: {ae}")
                    ASCIIColors.error(f"Failed to save image {i+1}. SDK Image object structure might have changed.")
                    trace_exception(ae)
                    continue
                except IOError as e:
                    self.app.error(f"Failed to write image {i+1} to file: {file_path}")
                    trace_exception(e)
                    continue
                except Exception as e: # Catch any other unexpected errors during saving/metadata processing
                    self.app.error(f"Unexpected error processing image {i+1}: {e}")
                    trace_exception(e)
                    continue


            if not results and num_images > 0: # Check if we expected images but got none saved
                self.app.error("Image generation API call was made, but no images were successfully processed or saved.")
            elif not results and num_images == 0: # Unlikely case, but handle it
                 self.app.info("Requested 0 images, no images generated or saved as expected.")
            return results

        # Specific error handling for Google Cloud / Vertex AI
        except google_auth_exceptions.GoogleAuthError as e:
             self.app.error("Authentication Error during Google Imagen generation!")
             self.app.error("Please ensure your Google Cloud credentials (ADC or Service Account) are valid and have Vertex AI permissions.")
             trace_exception(e)
             self.vertex_ai_initialized = False # Force re-check next time
             return []
        except Exception as e: # Catch other potential API errors (quota, invalid args, etc.)
            self.app.error("An error occurred during Google Imagen generation:")
            err_str = str(e).lower()
            # Use specific exception types if available from the SDK for better handling
            # Example: from google.api_core import exceptions as api_exceptions
            # except api_exceptions.PermissionDenied: ...
            # except api_exceptions.ResourceExhausted: ... (Quota)
            # except api_exceptions.InvalidArgument: ...
            if "permission denied" in err_str:
                 self.app.error("Permission Denied: Check IAM roles for your account/service account (e.g., 'Vertex AI User', 'Service Usage Consumer' on Vertex AI API).")
            elif "quota" in err_str or "resource exhausted" in err_str:
                 self.app.error("Quota Exceeded: Check your Vertex AI API quotas in the Google Cloud console for the specific model and region.")
            elif "invalid argument" in err_str:
                 self.app.error(f"Invalid Argument: Review parameters (prompt, negative prompt='{negative_prompt}', aspect ratio='{aspect_ratio}', guidance={guidance_scale}, model='{model_id}', seed={api_seed_param}, num_images={num_images}). Check API documentation.")
            elif "could not find model" in err_str or "not found" in err_str:
                 self.app.error(f"Model Not Found: Ensure the model ID '{model_id}' is correct and available in the '{location}' region for your project.")
            elif "api key not valid" in err_str: # Less common with ADC, but possible
                 self.app.error("API Key Invalid (if used): Check your API key configuration.")
            elif "must be logged in" in err_str or "authentication" in err_str or "credentials" in err_str:
                 self.app.error("Authentication Failed: Please re-run 'gcloud auth application-default login' or verify your GOOGLE_APPLICATION_CREDENTIALS setup.")
                 self.vertex_ai_initialized = False # Force re-check next time
            elif "connection aborted" in err_str or "connection error" in err_str or "network is unreachable" in err_str:
                 self.app.error("Network Error: Could not connect to Google Cloud API. Check your internet connection and firewall settings.")
            else:
                 self.app.error("An unexpected API error occurred.")

            trace_exception(e) # Log the full traceback for debugging
            return [] # Return empty list on failure


    def paint_from_images(
        self,
        positive_prompt: str,
        images: List[str], # Expecting list of file paths
        negative_prompt: str = "",
        sampler_name="Default",
        seed=None,
        scale=None,
        steps=None,
        width=None,
        height=None,
        output_folder=None,
        output_file_name=None
        ) -> List[Dict[str, Any]]:
        """
        Placeholder for Image variation, editing, or inpainting using Vertex AI Imagen.
        Requires specific Imagen editing/variation models and different API calls.
        """
        self.app.warning("Image-to-Image (variation/editing) using Google Imagen requires specific models (e.g., image-editing) and is NOT implemented in this binding.")
        self.app.warning("This function currently does nothing. Check Google Cloud documentation for Vertex AI image editing capabilities.")
        # To implement this, you would need to:
        # 1. Identify the correct Vertex AI model ID for editing/variation.
        # 2. Import the corresponding SDK class (e.g., ImageEditingModel, if available).
        # 3. Adapt the _initialize_vertex_ai or add a new method if needed.
        # 4. Read the input image(s) and potentially mask(s) into bytes.
        # 5. Call the appropriate SDK method (e.g., `edit_image(...)`) with parameters including image bytes, prompt, etc.
        # 6. Process the response and save the output images similar to the `paint` method.
        return [] # Return empty list as it's not implemented

    @staticmethod
    def get(app: LollmsApplication) -> 'LollmsGoogle':
        # Lollms likely handles instantiation and passes 'app' etc. to __init__.
        # Returning the class type is standard for such factory patterns if used.
        return LollmsGoogle