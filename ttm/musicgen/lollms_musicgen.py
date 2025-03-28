import importlib
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Attempt to import pipmaster. If it fails, provide instructions.
try:
    import pipmaster as pm
except ImportError:
    print("ERROR: pipmaster library not found.")
    print("Please install it using: pip install pipmaster")
    print("Lollms requires pipmaster to automatically install binding dependencies.")
    # You might want to raise an exception or exit here depending on how critical this is at import time.
    # For now, we'll let it proceed, but install/verify will fail later.
    raise("pipmaster is mandatory")

pm.install_if_missing("torch", index_url="https://download.pytorch.org/whl/cu124")
pm.install_if_missing("torchvision", index_url="https://download.pytorch.org/whl/cu124")
pm.install_if_missing("torchaudio", index_url="https://download.pytorch.org/whl/cu124")

from lollms.app import LollmsApplication
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.ttm import LollmsTTM  # Assuming the base class is in lollms.ttm
from lollms.helpers import ASCIIColors
from lollms.utilities import PackageManager, show_yes_no_dialog

# Helper function to check if Torch CUDA is available
def check_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# Configuration structure for MusicGenTTM
def_ult_cfg = {
    # ----- Model selection -----
    "model_name":"facebook/musicgen-small", # Or "facebook/musicgen-medium", "facebook/musicgen-large"
    # ----- Generation parameters -----
    "max_new_tokens": 256, # Corresponds roughly to duration. Adjust based on model and desired length. ~50 tokens/sec is a rough estimate.
    "guidance_scale": 3.0, # Higher values follow prompt more closely, lower values more creative. Only for relevant models.
    "temperature": 1.0, # Controls randomness. Higher values mean more randomness.
    "top_k": 250, # Limits sampling to the top K most likely tokens.
    "top_p": 0.0, # Nucleus sampling. 0 means disabled.
    "seed": -1, # Random seed for reproducibility. -1 for random.
    # ----- Hardware -----
    "device": "auto", # "auto", "cpu", "cuda", "mps" 
    # ----- Installation -----
    "always_update_transformers": False # Force update transformers library on init
}

class MusicGenTTMConfig(BaseConfig):
    def __init__(self, config:dict=def_ult_cfg):
        super().__init__(config)
    
    # You can add validation logic here if needed
    # For example, ensuring device is one of the allowed values

# The main binding class
class MusicGenTTM(LollmsTTM):
    """
    Lollms binding for Facebook's MusicGen Text-to-Music model using the transformers library.
    """
    def __init__(
                    self,
                    app: LollmsApplication,
                    config: Optional[dict]=None, # Allow passing config directly
                    service_config: Optional[TypedConfig] = None, # Accept TypedConfig
                    output_folder: Optional[str | Path] = None,
                    ):
        """
        Initializes the MusicGenTTM binding.

        Args:
            app (LollmsApplication): The Lollms application instance.
            config (dict, optional): A dictionary containing configuration options. Defaults to None.
            service_config (TypedConfig, optional): A TypedConfig object containing configuration options.
                                                   Overrides `config` if provided. Defaults to None.
            output_folder (str | Path, optional): Folder to save generated audio. Defaults to None (uses default Lollms output path).
        """
        # Get the configuration
        if service_config:
            cfg = service_config.config  # Extract dict from TypedConfig
        elif config:
            cfg = config
        else:
            cfg = {} # Will use defaults from template

        # Define the configuration template
        config_template = ConfigTemplate([
            # ----- Model selection -----
            {"name":"model_name",        "type":"str",    "value":"facebook/musicgen-small", "help":"The specific MusicGen model checkpoint to use (e.g., 'facebook/musicgen-small', 'facebook/musicgen-medium'). Larger models require more VRAM/RAM."},
            # ----- Generation parameters -----
            {"name":"max_new_tokens",    "type":"int",    "value":256,  "min":10, "help":"Maximum number of tokens to generate. This strongly influences the duration of the output audio. Roughly 50 tokens ~ 1 second."},
            {"name":"guidance_scale",    "type":"float",  "value":3.0,  "min":1.0, "help":"Controls how strictly the model follows the prompt (higher values) vs. being more creative (lower values)."},
            {"name":"temperature",       "type":"float",  "value":1.0,  "min":0.1, "max": 2.0, "help":"Controls the randomness of the output. Higher values increase randomness."},
            {"name":"top_k",             "type":"int",    "value":250,  "min":0,   "help":"Restricts sampling to the top K most probable tokens. 0 disables."},
            {"name":"top_p",             "type":"float",  "value":0.0,  "min":0.0, "max":1.0, "help":"Nucleus sampling threshold. 0 disables."},
            {"name":"seed",              "type":"int",    "value":-1,   "help":"Seed for random number generation. Use -1 for a random seed."},
            # ----- Hardware -----
            {"name":"device",            "type":"str",    "value":"auto", "help":"Device to run the model on ('auto', 'cpu', 'cuda', 'mps'). 'auto' tries CUDA/MPS first, then CPU."},
            # ----- Installation -----
            {"name":"always_update_transformers", "type":"bool", "value":False, "help":"If checked, the transformers library will be updated every time the binding is loaded."}
        ])
        
        # Create the TypedConfig
        typed_config = TypedConfig(config_template, cfg)

        # Initialize the base LollmsSERVICE class
        super().__init__(
            name="musicgen", # Unique name for the service
            app=app,
            service_config=typed_config,
            output_folder=output_folder
        )

        self.model = None
        self.processor = None
        self.torch = None
        self.transformers = None
        self.scipy = None
        self._device = None # Internal device storage

        # Install dependencies immediately if requested or not verified
        # This assumes install() uses pipmaster correctly
        if not self.verify(app):
             # Use PackageManager for installation prompts
            if show_yes_no_dialog("Confirmation", f"The MusicGen binding requires installing dependencies (torch, transformers, scipy). Install now?"):
                self.install(app)
            else:
                self.app.error("MusicGen binding dependencies not installed. The binding may not function correctly.")
                return # Or raise an error

        # Optionally force update transformers if configured
        if self.config.always_update_transformers and pm:
            self.app.ShowBlockingMessage("Updating transformers library...")
            pm.install_if_missing("transformers", always_update=True)
            self.app.HideBlockingMessage()


    @staticmethod
    def verify(app: LollmsApplication) -> bool:
        """
        Verifies if the necessary libraries (torch, transformers, scipy) are installed.
        """
        required = ["torch", "transformers", "scipy"]
        missing = []
        for package in required:
            try:
                importlib.import_module(package)
            except ImportError:
                missing.append(package)

        if missing:
            app.WarningMessage(f"MusicGen binding verification failed. Missing packages: {', '.join(missing)}")
            return False
        else:
            app.InfoMessage("MusicGen binding verified successfully.")
            return True

    @staticmethod
    def install(app: LollmsApplication) -> bool:
        """
        Installs the necessary libraries using pipmaster.
        """
        if not pm:
            app.error("pipmaster library is required for automatic installation but is not available.")
            return False
            
        app.ShowBlockingMessage("Installing MusicGen dependencies (torch, transformers, scipy). This might take a while...")
        
        try:
            # Install PyTorch (handle CPU/CUDA versions carefully if needed, pipmaster might simplify this)
            # For simplicity, let pipmaster find a suitable torch version.
            # More specific control might be needed for CUDA vs CPU if auto-detection fails.
            pm.install_if_missing("torch") 
            
            # Install transformers
            pm.install_if_missing("transformers")
            
            # Install SciPy (for saving WAV files)
            pm.install_if_missing("scipy")

            # Optional: Install accelerate for potentially better performance/memory usage on larger models
            # pm.install_if_missing("accelerate") 

            app.InfoMessage("MusicGen dependencies installed successfully.")
            app.HideBlockingMessage()
            # Force verification after install
            PackageManager.rebuild_packages()
            return True
        
        except Exception as e:
            app.error(f"Failed to install MusicGen dependencies: {e}")
            app.HideBlockingMessage()
            return False

    def _load_model(self):
        """Loads the MusicGen model and processor."""
        if self.model is not None and self.processor is not None:
            return True # Already loaded

        self.app.ShowBlockingMessage(f"Loading MusicGen model: {self.config.model_name}...")
        try:
            # Import necessary libraries (only when needed)
            self.torch = importlib.import_module("torch")
            self.transformers = importlib.import_module("transformers")
            self.scipy = importlib.import_module("scipy.io.wavfile")

            # Determine device
            if self.config.device == "auto":
                if check_torch_cuda():
                    self._device = "cuda"
                # Check for MPS (Apple Silicon) - Requires recent torch/transformers
                # elif self.torch.backends.mps.is_available():
                #     self._device = "mps"
                else:
                    self._device = "cpu"
            else:
                self._device = self.config.device
            
            self.app.InfoMessage(f"MusicGen using device: {self._device}")

            # Load processor and model
            self.processor = self.transformers.AutoProcessor.from_pretrained(self.config.model_name)
            self.model = self.transformers.MusicgenForConditionalGeneration.from_pretrained(self.config.model_name)
            
            # Move model to device
            self.model.to(self._device)
            
            self.app.InfoMessage(f"MusicGen model '{self.config.model_name}' loaded successfully.")
            self.app.HideBlockingMessage()
            return True

        except Exception as e:
            self.app.error(f"Failed to load MusicGen model: {e}")
            self.app.HideBlockingMessage()
            self.model = None
            self.processor = None
            self._device = None
            return False


    def generate(self,
                 prompt: str,
                 negative_prompt: str = "", # MusicGen doesn't typically use negative prompts this way, but keep for base class consistency
                 duration_s: Optional[float] = None, # Override default duration via tokens
                 seed: Optional[int] = None,
                 guidance_scale: Optional[float] = None,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 max_new_tokens: Optional[int] = None,
                 output_dir: Optional[str | Path] = None,
                 output_file_name: Optional[str] = None
                 ) -> List[Dict[str, str]]:
        """
        Generates audio based on the given text prompt using MusicGen.

        Args:
            prompt (str): The positive prompt describing the desired music.
            negative_prompt (str, optional): A prompt describing elements to avoid (largely ignored by MusicGen but kept for interface). Defaults to "".
            duration_s (float, optional): Desired duration in seconds. Overrides `max_new_tokens` from config if provided. Defaults to None.
            seed (int, optional): Seed for reproducibility. Overrides config seed if provided. Defaults to None.
            guidance_scale (float, optional): Overrides config guidance_scale. Defaults to None.
            temperature (float, optional): Overrides config temperature. Defaults to None.
            top_k (int, optional): Overrides config top_k. Defaults to None.
            top_p (float, optional): Overrides config top_p. Defaults to None.
            max_new_tokens (int, optional): Overrides config max_new_tokens. Ignored if `duration_s` is set. Defaults to None.
            output_dir (str | Path, optional): Directory to save the output file(s). If None, uses self.output_folder.
            output_file_name (str, optional): Desired name for the output file (without extension). If None, a unique name will be generated.

        Returns:
            List[Dict[str, str]]: A list containing one dictionary with details about the generated audio file.
                                  Keys: 'path', 'url', 'prompt', 'duration_s' (estimated), 'seed', 'format'. Returns empty list on failure.
        """
        if not self._load_model():
            self.app.error("MusicGen model not loaded. Cannot generate audio.")
            return []

        self.app.ShowBlockingMessage(f"Generating music for prompt: '{prompt[:50]}...'")

        try:
            # --- Parameter Preparation ---
            gen_seed = seed if seed is not None else self.config.seed
            gen_guidance_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale
            gen_temperature = temperature if temperature is not None else self.config.temperature
            gen_top_k = top_k if top_k is not None else self.config.top_k
            gen_top_p = top_p if top_p is not None else self.config.top_p
            
            if duration_s is not None and duration_s > 0:
                # Estimate tokens based on duration (this is approximate!)
                # MusicGen models typically have a sampling rate (e.g., 32000 Hz) and frame rate for the encoder/decoder.
                # A common rough estimate is ~50 tokens per second for the decoder.
                sampling_rate = self.model.config.audio_encoder.sampling_rate
                tokens_per_second = 50 # Adjust if models differ significantly
                gen_max_new_tokens = int(duration_s * tokens_per_second)
                self.app.InfoMessage(f"Duration set to {duration_s}s, estimated max_new_tokens: {gen_max_new_tokens}")
            else:
                 gen_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens

            # Ensure max_new_tokens is reasonable
            if gen_max_new_tokens <= 0:
                gen_max_new_tokens = 10 # Set a minimum
                self.app.WarningMessage(f"max_new_tokens was too low, adjusted to {gen_max_new_tokens}.")


            # Prepare inputs
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            # Setup generator for seed
            generator = None
            if gen_seed != -1:
                generator = self.torch.Generator(device=self._device).manual_seed(gen_seed)

            # --- Generation ---
            with self.torch.no_grad(): # Ensure gradients aren't calculated
                 # Note: MusicGen doesn't directly use negative_prompt in generation args like Stable Diffusion
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_max_new_tokens,
                    guidance_scale=gen_guidance_scale,
                    temperature=gen_temperature,
                    top_k=gen_top_k,
                    top_p=gen_top_p,
                    do_sample=True, # Essential for music generation
                    generator=generator
                    # Add other relevant parameters if supported by the specific transformers version/model
                )[0] # Get the first (and usually only) audio output in the batch

            # --- Post-processing and Saving ---
            audio_numpy = audio_values.cpu().numpy().squeeze() # Remove batch dim and move to CPU
            sampling_rate = self.model.config.audio_encoder.sampling_rate

            # Determine output path
            output_path = Path(output_dir or self.output_folder)
            output_path.mkdir(parents=True, exist_ok=True)

            if output_file_name:
                base_filename = output_file_name
            else:
                # Generate a unique filename
                import time
                timestamp = int(time.time())
                safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
                base_filename = f"musicgen_{safe_prompt}_{timestamp}"
                
            output_filepath = (output_path / f"{base_filename}.wav").resolve()
            
            # Save as WAV file
            # Ensure data is in suitable format for wavfile.write (usually float32 in [-1, 1] or int16)
            # MusicGen output is typically float32 in [-1, 1]
            if audio_numpy.dtype != self.np.float32:
                 audio_numpy = audio_numpy.astype(self.np.float32)
            # Normalize just in case it exceeds [-1, 1] slightly
            # max_val = np.max(np.abs(audio_numpy))
            # if max_val > 1.0:
            #     audio_numpy = audio_numpy / max_val

            self.scipy.wavfile.write(output_filepath, rate=sampling_rate, data=audio_numpy)

            # Calculate actual duration
            actual_duration_s = len(audio_numpy) / sampling_rate

            self.app.InfoMessage(f"Music generated successfully and saved to: {output_filepath}")

            # --- Prepare Result ---
            result = {
                "path": str(output_filepath),
                "url": f"/outputs/{output_filepath.relative_to(self.app.lollms_paths.personal_outputs_path)}", # URL relative to outputs route
                "prompt": prompt,
                # "negative_prompt": negative_prompt, # Keep if needed downstream, but wasn't used for generation
                "duration_s": round(actual_duration_s, 2),
                "seed": gen_seed,
                "guidance_scale": gen_guidance_scale,
                "temperature": gen_temperature,
                "top_k": gen_top_k,
                "top_p": gen_top_p,
                "max_new_tokens_used": gen_max_new_tokens, # Parameter used, actual tokens might differ
                "format": "wav",
                "model": self.config.model_name
            }
            
            self.app.HideBlockingMessage()
            return [result]

        except Exception as e:
            self.app.error(f"Error generating music with MusicGen: {e}")
            # Log the full traceback for debugging
            import traceback
            self.app.error(traceback.format_exc()) 
            self.app.HideBlockingMessage()
            return []

    @staticmethod
    def get(app: LollmsApplication) -> 'MusicGenTTM':
        """ Returns the MusicGenTTM class type. """
        return MusicGenTTM