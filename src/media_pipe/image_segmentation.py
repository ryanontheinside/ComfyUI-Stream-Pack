import numpy as np
import mediapipe as mp
import torch
import os
import cv2 # Added for potential debugging/visualization if needed later
import urllib.request # Added for model downloader
import hashlib # Added for potential future checksumming (optional)
import platform # Added for OS check
import folder_paths # Added for ComfyUI model paths

# --- REMOVED Path Registration Block ---
# (No code here - registration happens at runtime now)

# Helper function to get the directory of the current script
def get_script_directory():
    return os.path.dirname(os.path.realpath(__file__))


def get_mediapipe_models_directory():
    base_path = folder_paths.models_dir
    mediapipe_dir = os.path.join(base_path, "mediapipe", "segmentation")
    os.makedirs(mediapipe_dir, exist_ok=True)
    return mediapipe_dir

class MediaPipeImageSegmentationNode:
    """
    A ComfyUI node for real-time image segmentation using MediaPipe.
    Optimized for streaming by initializing the segmenter once.
    Accepts model information from a dedicated loader node.
    Outputs a segmentation mask (typically foreground).
    """
    def __init__(self):
        self.segmenter = None
        self.current_model_path = None
        self.current_output_confidence_masks = None # Track this setting too

    @classmethod
    def INPUT_TYPES(cls):
        # Model path is now handled by the loader node
        return {
            "required": {
                "image": ("IMAGE",),
                "model_info": ("MEDIAPIPE_MODEL_INFO",), # Accept info from loader
                 "output_confidence_masks": ("BOOLEAN", {
                    "default": False,
                    "description": "Output confidence masks (float 0-1, primary foreground) instead of a binary category mask."
                }),
                 "threshold": ("FLOAT", {
                     "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                     "display": "slider",
                     "description": "Confidence threshold (only used when output_confidence_masks=True)."
                 }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment_image"
    CATEGORY = "ComfyUI-Stream-Pack/MediaPipe" # Match potential existing category

    # Function to initialize or update the segmenter if settings change
    def _initialize_segmenter(self, model_path, output_confidence_masks):
        # Model path is now validated and passed directly from the loader node.
        # No need for relative path resolution here anymore.

        # Check if re-initialization is needed
        if (self.segmenter is None or
            self.current_model_path != model_path or # Use the direct path
            self.current_output_confidence_masks != output_confidence_masks):

            if not model_path or not os.path.exists(model_path):
                 # This check should ideally not fail if the loader node worked correctly, but good practice.
                 raise FileNotFoundError(f"Model file path received from loader is invalid or file doesn't exist: '{model_path}'.")

            print(f"[MediaPipeImageSegmentationNode] Initializing MediaPipe Image Segmenter...")
            print(f"  Model: {model_path}")
            print(f"  Output Confidence Masks: {output_confidence_masks}")

            BaseOptions = mp.tasks.BaseOptions
            ImageSegmenter = mp.tasks.vision.ImageSegmenter
            ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            try:
                # Determine delegate based on availability and platform
                delegate = BaseOptions.Delegate.CPU # Default to CPU
                if platform.system().lower() != 'windows':
                    if hasattr(BaseOptions.Delegate, 'GPU'):
                        delegate = BaseOptions.Delegate.GPU
                        print("  Attempting to use GPU delegate (non-Windows platform).")
                    else:
                        print("  GPU delegate not available on this platform, using CPU.")
                else:
                    print("  Windows platform detected, forcing CPU delegate.")

                print(f"  Using Delegate: {delegate.name}")

                options = ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
                    running_mode=VisionRunningMode.IMAGE, # Process image by image
                    output_category_mask=not output_confidence_masks, # Request category mask if not requesting confidence masks
                    output_confidence_masks=output_confidence_masks
                )
                # Close existing segmenter before creating a new one
                if self.segmenter:
                    self.segmenter.close()

                self.segmenter = ImageSegmenter.create_from_options(options)
                self.current_model_path = model_path
                self.current_output_confidence_masks = output_confidence_masks
                print(f"[MediaPipeImageSegmentationNode] MediaPipe Image Segmenter initialized successfully.")
            except Exception as e:
                # Clean up state if initialization fails
                if self.segmenter:
                    self.segmenter.close()
                self.segmenter = None
                self.current_model_path = None
                self.current_output_confidence_masks = None
                # Add more specific error info if possible
                detailed_error = f"Failed to initialize MediaPipe Image Segmenter: {e}"
                if "GPU Delegate is not yet supported" in str(e):
                     detailed_error += " (Note: GPU acceleration for MediaPipe tasks might not be available on your OS/hardware)"
                raise RuntimeError(detailed_error)

    # Updated function signature to accept model_info
    def segment_image(self, image: torch.Tensor, model_info: dict, output_confidence_masks: bool, threshold: float):
        # Extract model path from the info dictionary
        model_path = model_info.get("model_path")
        if not model_path:
             print("[MediaPipeImageSegmentationNode] Error: Invalid model_info received (missing 'model_path'). Returning zero mask.")
             batch_size, height, width, _ = image.shape
             return (torch.zeros((batch_size, height, width), dtype=torch.float32),)

        # Ensure segmenter is initialized with the correct settings
        try:
             self._initialize_segmenter(model_path, output_confidence_masks)
        except Exception as e:
             print(f"[MediaPipeImageSegmentationNode] Error initializing segmenter: {e}")
             batch_size, height, width, _ = image.shape
             return (torch.zeros((batch_size, height, width), dtype=torch.float32),)

        if self.segmenter is None:
             print("[MediaPipeImageSegmentationNode] Error: MediaPipe Image Segmenter is not initialized. Returning zero mask.")
             batch_size, height, width, _ = image.shape
             return (torch.zeros((batch_size, height, width), dtype=torch.float32),)

        batch_size, height, width, _ = image.shape
        output_masks_list = []

        for i in range(batch_size):
            img_slice_np = image[i].numpy()
            if img_slice_np.dtype != np.float32 and img_slice_np.dtype != np.float64:
                 img_slice_np = img_slice_np.astype(np.float32)
            img_rgb_uint8 = (img_slice_np * 255).astype(np.uint8)
            img_rgb_uint8_cont = np.ascontiguousarray(img_rgb_uint8)

            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_uint8_cont)
            except Exception as e:
                 print(f"[MediaPipeImageSegmentationNode] Error creating MediaPipe Image object for image {i}: {e}. Returning zero mask for this item.")
                 mask_np = np.zeros((height, width), dtype=np.float32)
                 output_masks_list.append(torch.from_numpy(mask_np))
                 continue

            try:
                 segmentation_result = self.segmenter.segment(mp_image)
            except Exception as e:
                 print(f"[MediaPipeImageSegmentationNode] Error during segmentation for image {i}: {e}. Returning zero mask for this item.")
                 mask_np = np.zeros((height, width), dtype=np.float32)
                 output_masks_list.append(torch.from_numpy(mask_np))
                 continue

            mask_np = np.zeros((height, width), dtype=np.float32)
            if output_confidence_masks:
                if segmentation_result.confidence_masks is not None and len(segmentation_result.confidence_masks) > 0:
                     target_mask_index = 1 if len(segmentation_result.confidence_masks) > 1 else 0
                     primary_mask = segmentation_result.confidence_masks[target_mask_index].numpy_view()
                     mask_np = np.where(primary_mask >= threshold, primary_mask, 0.0).astype(np.float32)
                else:
                     print(f"[MediaPipeImageSegmentationNode] Warning: output_confidence_masks is True, but no confidence masks found for image {i}. Returning zero mask.")

            else: # Output category mask
                if segmentation_result.category_mask is not None:
                    category_mask_np = segmentation_result.category_mask.numpy_view()
                    mask_np = (category_mask_np > 0).astype(np.float32)
                else:
                    print(f"[MediaPipeImageSegmentationNode] Warning: output_confidence_masks is False, but no category mask found for image {i}. Returning zero mask.")

            mask_tensor = torch.from_numpy(mask_np)
            output_masks_list.append(mask_tensor)

        if not output_masks_list:
             print("[MediaPipeImageSegmentationNode] Warning: No masks generated. Returning zero tensor.")
             return (torch.zeros((batch_size, height, width), dtype=torch.float32),)

        first_shape = output_masks_list[0].shape
        if not all(t.shape == first_shape for t in output_masks_list):
             print("[MediaPipeImageSegmentationNode] Error: Inconsistent mask shapes. Returning zero mask.")
             return (torch.zeros((batch_size, height, width), dtype=torch.float32),)

        try:
            output_batch = torch.stack(output_masks_list, dim=0)
            return (output_batch,)
        except Exception as e:
             print(f"[MediaPipeImageSegmentationNode] Error stacking masks: {e}. Returning zero mask.")
             return (torch.zeros((batch_size, height, width), dtype=torch.float32),)

    # Optional: Clean up the MediaPipe segmenter when the node is deleted or ComfyUI shuts down
    def __del__(self):
        if hasattr(self, 'segmenter') and self.segmenter:
            try:
                print("[MediaPipeImageSegmentationNode] Closing MediaPipe Image Segmenter...")
                self.segmenter.close()
            except Exception as e:
                print(f"[MediaPipeImageSegmentationNode] Error closing MediaPipe Image Segmenter: {e}")


# --- Model Loader Node ---

# Define available models and their URLs
# Using common names and expected filenames
AVAILABLE_MODELS = {
    "DeepLabV3 (General Purpose)": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite", # <-- Specific version URL
        "filename": "deeplab_v3.tflite"
    },
    "Selfie Segmentation (Landscape)": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float32/latest/selfie_segmenter_landscape.tflite",
        "filename": "selfie_segmenter_landscape.tflite"
    },
    "Selfie Segmentation (General)": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float32/latest/selfie_segmenter.tflite",
        "filename": "selfie_segmenter.tflite"
    },
    # Add more models here as needed
}


class MediaPipeModelLoaderNode:
    """
    A ComfyUI node to select, download (if necessary), and provide
    the path to a MediaPipe segmentation model saved in ComfyUI's models directory.
    Handles path resolution and download only during node execution.
    """
    def __init__(self):
        # Do **NOTHING** related to paths here.
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(AVAILABLE_MODELS.keys()), {
                    "default": list(AVAILABLE_MODELS.keys())[0]
                }),
            }
        }

    RETURN_TYPES = ("MEDIAPIPE_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "load_model"
    CATEGORY = "ComfyUI-Stream-Pack/MediaPipe"

    def download_model(self, url, save_path):
        """Downloads a file from a URL to a specified path."""
        print(f"[MediaPipeModelLoaderNode] Downloading model from {url} to {save_path}...")
        try:
            # Ensure parent directory exists before download attempt
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
                total_size_header = response.getheader('Content-Length')
                total_size = int(total_size_header) if total_size_header else None

                if total_size:
                    print(f"  File size: {total_size / (1024*1024):.2f} MB")
                else:
                    print("  File size unknown.")

                block_size = 8192 # Good default block size
                bytes_downloaded = 0
                last_print_bytes = 0
                # Print progress more granularly, e.g., every MB or few percent
                print_interval = 1024 * 1024 # Print every ~1MB

                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    bytes_downloaded += len(buffer)

                    # Progress reporting logic
                    if total_size:
                         progress = (bytes_downloaded / total_size) * 100
                         if bytes_downloaded - last_print_bytes > print_interval or bytes_downloaded == total_size:
                            print(f"  Downloaded: {bytes_downloaded / (1024*1024):.2f} / {total_size / (1024*1024):.2f} MB ({progress:.1f}%)", end='                 \\r')
                            last_print_bytes = bytes_downloaded
                    elif bytes_downloaded - last_print_bytes > print_interval:
                         print(f"  Downloaded: {bytes_downloaded / (1024*1024):.2f} MB", end='                 \\r')
                         last_print_bytes = bytes_downloaded

            # Newline after download progress
            print() # Ensures the "Download complete" message is on a new line
            print(f"[MediaPipeModelLoaderNode] Download complete: {save_path}")
            return True
        except Exception as e:
            print(f"\n[MediaPipeModelLoaderNode] Error downloading model: {e}") # Newline before error
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    print(f"[MediaPipeModelLoaderNode] Removed incomplete download: {save_path}")
                except Exception as rm_err:
                    print(f"[MediaPipeModelLoaderNode] Error removing incomplete download {save_path}: {rm_err}")
            return False

    def load_model(self, model_name):
        # --- Runtime Path Handling --- 
        # 1. Get the mediapipe models directory NOW
        models_dir = get_mediapipe_models_directory()
        print(f"[MediaPipeModelLoaderNode] Using models directory: {models_dir}") # Log the used path

        # 2. Validate selected model name
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined in AVAILABLE_MODELS.")

        # 3. Get model details
        model_details = AVAILABLE_MODELS[model_name]
        model_filename = model_details["filename"]
        model_url = model_details["url"]
        
        # 4. Construct the full path
        model_path = os.path.join(models_dir, model_filename)

        # 5. Check if model exists, download if not
        if not os.path.exists(model_path):
            print(f"[MediaPipeModelLoaderNode] Model '{model_filename}' not found locally at {model_path}.")
            if not self.download_model(model_url, model_path):
                 raise RuntimeError(f"Failed to download model '{model_filename}'. Check URL ({model_url}) and network connection.")
        else:
             print(f"[MediaPipeModelLoaderNode] Found model '{model_filename}' at {model_path}")

        # 6. Return the model info
        model_info = {
            "model_path": model_path,
            "model_name": model_name
        }
        return (model_info,)

# --- Node Mappings --- (Remain the same)
NODE_CLASS_MAPPINGS = {
    "MediaPipeImageSegmentationNode": MediaPipeImageSegmentationNode,
    "MediaPipeModelLoaderNode": MediaPipeModelLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeImageSegmentationNode": "MediaPipe Image Segmentation",
    "MediaPipeModelLoaderNode": "MediaPipe Model Loader",
}
