"""
Lollms TTM Module
=================

This module is part of the Lollms library, designed to provide Text-to-Music (TTM) functionalities within the LollmsApplication framework. The base class `LollmsTTM` is intended to be inherited and implemented by other classes that provide specific TTM functionalities.

Author: ParisNeo, a computer geek passionate about AI
"""

from lollms.app import LollmsApplication
from pathlib import Path
from typing import List, Dict
from lollms.ttm import LollmsTTM
from lollms.utilities import PackageManager, File_Path_Generator, check_and_install_torch

class LollmsMusicGen(LollmsTTM):
    """
    LollmsMusicGen is a model class for implementing Text-to-Music (TTM) functionalities within the LollmsApplication.
    
    Attributes:
        app (LollmsApplication): The instance of the main Lollms application.
        model (str): The TTM model to be used for image generation.
        api_key (str): API key for accessing external TTM services (if needed).
        output_path (Path or str): Path where the output image files will be saved.
        voices (List[str]): List of available voices for TTM (to be filled by the child class).
        models (List[str]): List of available models for TTM (to be filled by the child class).
    """
    
    def __init__(
                    self,
                    name:str,
                    app: LollmsApplication, 
                    model="facebook/musicgen-melody",#"facebook/musicgen-small","facebook/musicgen-medium","facebook/musicgen-melody","facebook/musicgen-large"
                    device="cuda",
                    api_key="",
                    output_path=None
                    ):
        """
        Initializes the LollmsTTM class with the given parameters.

        Args:
            app (LollmsApplication): The instance of the main Lollms application.
            model (str, optional): The TTM model to be used for image generation. Defaults to an empty string.
            api_key (str, optional): API key for accessing external TTM services. Defaults to an empty string.
            output_path (Path or str, optional): Path where the output image files will be saved. Defaults to None.
        """
        self.name = name
        self.app = app
        self.model = model
        self.api_key = api_key
        self.output_path = output_path
        if not PackageManager.check_package_installed("musicgen"):
            check_and_install_torch(True if device=="cuda" else False)
            PackageManager.install_or_update("musicgen")

        from audiocraft.models import musicgen
        self.music_model = musicgen.MusicGen.get_pretrained(model, device=device)

        self.models = [] # To be filled by the child class
        self.ready = True

    def generate(self, 
                positive_prompt: str, 
                negative_prompt: str = "",
                duration=30,
                generation_engine=None,
                output_path = None) -> List[Dict[str, str]]:
        """
        Generates images based on the given positive and negative prompts.

        Args:
            positive_prompt (str): The positive prompt describing the desired image.
            negative_prompt (str, optional): The negative prompt describing what should be avoided in the image. Defaults to an empty string.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing image paths, URLs, and metadata.
        """
        if output_path is None:
            output_path = self.output_path
        import torchaudio
        self.music_model.set_generation_params(duration=duration)
        res = self.music_model.generate([positive_prompt], progress=True)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = File_Path_Generator.generate_unique_file_path(output_path, "generation","wav")
        torchaudio.save(output_file, res.reshape(1, -1).cpu(), 32000)

        return  output_file, {"prompt":positive_prompt,"duration":duration}

    def generate_from_samples(self, positive_prompt: str, samples: List[str], negative_prompt: str = "") -> List[Dict[str, str]]:
        """
        Generates images based on the given positive prompt and reference images.

        Args:
            positive_prompt (str): The positive prompt describing the desired image.
            images (List[str]): A list of paths to reference images.
            negative_prompt (str, optional): The negative prompt describing what should be avoided in the image. Defaults to an empty string.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing image paths, URLs, and metadata.
        """
        pass

    @staticmethod
    def verify(app: LollmsApplication) -> bool:
        """
        Verifies if the TTM service is available.

        Args:
            app (LollmsApplication): The instance of the main Lollms application.

        Returns:
            bool: True if the service is available, False otherwise.
        """
        return True

    @staticmethod
    def install(app: LollmsApplication) -> bool:
        """
        Installs the necessary components for the TTM service.

        Args:
            app (LollmsApplication): The instance of the main Lollms application.

        Returns:
            bool: True if the installation was successful, False otherwise.
        """
        return True
    
    @staticmethod 
    def get(app: LollmsApplication) -> 'LollmsTTM':
        """
        Returns the LollmsTTM class.

        Args:
            app (LollmsApplication): The instance of the main Lollms application.

        Returns:
            LollmsTTM: The LollmsTTM class.
        """
        return LollmsTTM
