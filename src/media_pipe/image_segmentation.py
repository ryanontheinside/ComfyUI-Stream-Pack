import numpy as np
import mediapipe as mp
import torch
import os
import urllib.request # Added for model downloader
import platform # Added for OS check
import folder_paths # Added for ComfyUI model paths


# Helper function to get the directory of the current script
def get_script_directory():
    return os.path.dirname(os.path.realpath(__file__))


def get_mediapipe_models_directory():
    base_path = folder_paths.models_dir
    mediapipe_dir = os.path.join(base_path, "mediapipe", "segmentation")
    os.makedirs(mediapipe_dir, exist_ok=True)
    return mediapipe_dir

# Define class names and their corresponding indices for multiclass models
MULTICLASS_NAMES = {
    "Background": 0,
    "Hair": 1,
    "Body-skin": 2,
    "Face-skin": 3,
    "Clothes": 4,
    "Accessories/Other": 5,
}

# Uses MULTICLASS_NAMES indices
CLASS_COLORS_BGR = {
    MULTICLASS_NAMES["Background"]: (0, 0, 0),       # Black
    MULTICLASS_NAMES["Hair"]: (255, 0, 0),     # Blue
    MULTICLASS_NAMES["Body-skin"]: (0, 255, 255),   # Yellow
    MULTICLASS_NAMES["Face-skin"]: (255, 200, 100), # Light Blue
    MULTICLASS_NAMES["Clothes"]: (0, 255, 0),     # Green
    MULTICLASS_NAMES["Accessories/Other"]: (255, 0, 255),   # Magenta
}

# Define available models and their URLs
# Using common names and expected filenames
AVAILABLE_MODELS = {
    "DeepLabV3 (General Purpose)": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite", # <-- Specific version URL
        "filename": "deeplab_v3.tflite"
    },
    "Selfie Segmentation (Landscape)": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
        "filename": "selfie_segmenter_landscape.tflite"
    },
    "Selfie Segmentation (General)": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        "filename": "selfie_segmenter.tflite"
    },
    "Hair Segmentation": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
        "filename": "hair_segmenter.tflite"
    },
    "Selfie Multiclass": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        "filename": "selfie_multiclass_256x256.tflite"
    },
    # Add more models here as needed
    # interactive segmentation, like sam2....
    # https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite
    # https://ai.google.dev/edge/mediapipe/solutions/vision/interactive_segmenter?utm_source=chatgpt.com

}

class MediaPipeModelLoaderNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(AVAILABLE_MODELS.keys()), {
                    "default": list(AVAILABLE_MODELS.keys())[0],
                    "tooltip": "Select the MediaPipe segmentation model to use. Different models are optimized for different tasks: DeepLabV3 for general segmentation, Selfie for person/background, Hair for hair segmentation, and Multiclass for detailed person segmentation."
                }),
            }
        }

    RETURN_TYPES = ("MEDIAPIPE_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "load_model"
    CATEGORY = "StreamPack/Segmentation"
    DESCRIPTION = "Loads a MediaPipe segmentation model and provides model information to the segmentation node. Downloads the model automatically if needed."

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

                block_size = 8192
                bytes_downloaded = 0
                last_print_bytes = 0
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
                         # Update progress bar less frequently to avoid slowing down download
                         if bytes_downloaded - last_print_bytes > print_interval or bytes_downloaded == total_size:
                            # Use carriage return `\r` to overwrite the line
                            print(f"  Downloaded: {bytes_downloaded / (1024*1024):.2f} / {total_size / (1024*1024):.2f} MB ({progress:.1f}%)", end='                 \r')
                            last_print_bytes = bytes_downloaded
                    elif bytes_downloaded - last_print_bytes > print_interval:
                         # Fallback for unknown size
                         print(f"  Downloaded: {bytes_downloaded / (1024*1024):.2f} MB", end='                 \r')
                         last_print_bytes = bytes_downloaded

            print()
            print(f"[MediaPipeModelLoaderNode] Download complete: {save_path}")
            return True
        except Exception as e:
            print(f"[MediaPipeModelLoaderNode] Error downloading model: {e}")
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    print(f"[MediaPipeModelLoaderNode] Removed incomplete download: {save_path}")
                except Exception as rm_err:
                    print(f"[MediaPipeModelLoaderNode] Error removing incomplete download {save_path}: {rm_err}")
            return False

    def load_model(self, model_name):
        # get the mediapipe models directory
        models_dir = get_mediapipe_models_directory()
        print(f"[MediaPipeModelLoaderNode] Using models directory: {models_dir}") # Log the used path

        # Validate selected model name
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined in AVAILABLE_MODELS.")

        #Get model details
        model_details = AVAILABLE_MODELS[model_name]
        model_filename = model_details["filename"]
        model_url = model_details["url"]

        # Construct the full path
        model_path = os.path.join(models_dir, model_filename)

        # Check if model exists, download if not
        if not os.path.exists(model_path):
            print(f"[MediaPipeModelLoaderNode] Model '{model_filename}' not found locally at {model_path}.")
            if not self.download_model(model_url, model_path):
                 raise RuntimeError(f"Failed to download model '{model_filename}'. Check URL ({model_url}) and network connection.")
        else:
             print(f"[MediaPipeModelLoaderNode] Found model '{model_filename}' at {model_path}")

        # Return the model info
        model_info = {
            "model_path": model_path,
            "model_name": model_name # Pass the name along
        }
        return (model_info,)


class MediaPipeImageSegmentationNode:

#TODO: sort out GPU delegate issues with mutliclass model

    def __init__(self):
        self.segmenter = None
        self.current_model_path = None
        self.current_output_confidence_masks = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image(s) to perform segmentation on. Batch processing is supported."
                }),
                "model_info": ("MEDIAPIPE_MODEL_INFO", {
                    "tooltip": "Information about the MediaPipe model to use, provided by the MediaPipe Model Loader node."
                }),
                "output_confidence_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, outputs confidence values (0.0-1.0) instead of binary category masks. Useful for soft masks with gradual transitions. Disables multiclass segment output."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Confidence threshold for filtering mask values. When output_confidence_masks=True, only pixels with confidence >= threshold retain their values, others become 0. When output_confidence_masks=False, this has no effect as MediaPipe internally uses a fixed threshold to generate category masks."
                }),
                "generate_visualization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, generates a colored visualization image showing the segmentation results. For multiclass models, each class gets a distinct color."
                }),
                "delegate_mode": (["cpu", "gpu"], {
                    "default": "cpu",
                    "tooltip": "Computation delegate to use. 'cpu' works on all platforms. 'gpu' may be faster but is not supported on Windows and requires GPU."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MP_INT_MASK",)
    RETURN_NAMES = ("mask", "visualization", "multiclass_segments",)
    FUNCTION = "segment_image"
    CATEGORY = "StreamPack/Segmentation"
    DESCRIPTION = """Performs image segmentation using MediaPipe models.

Output Types:
- mask: Binary mask (Selfie models, non-confidence mode), Confidence mask (if enabled), or Hair mask (Hair model).
- visualization: Optional colored image showing segmentation results.
- multiclass_segments: Raw integer mask containing class IDs (Selfie Multiclass model, non-confidence mode only).

Multiclass Segmentation Classes (for visualization and multiclass_segments):
- 0: Background        (Black, RGB: 0,0,0)
- 1: Hair              (Blue, RGB: 0,0,255)
- 2: Body-skin         (Yellow, RGB: 255,255,0)
- 3: Face-skin         (Light Blue, RGB: 100,200,255)
- 4: Clothes           (Green, RGB: 0,255,0)
- 5: Accessories/Other (Magenta, RGB: 255,0,255)

Threshold Usage:
- In confidence mode (output_confidence_masks=True): Threshold controls which pixels appear in the output mask.
- In category mode (output_confidence_masks=False): Threshold has no effect as MediaPipe internally uses fixed thresholds.

Enable 'generate_visualization' to see colored results. Use the 'Select MediaPipe Segment' node to extract specific masks from 'multiclass_segments'."""

    # Function to initialize or update the segmenter if settings change
    def _initialize_segmenter(self, model_path, output_confidence_masks, delegate_mode="cpu"):
        # Check if re-initialization is needed (model path or output mode changed, or segmenter doesn't exist)
        if (self.segmenter is None or
            self.current_model_path != model_path or
            self.current_output_confidence_masks != output_confidence_masks):

            # Close the existing segmenter *before* creating a new one if it exists
            if hasattr(self, 'segmenter') and self.segmenter:
                try:
                    print("[MediaPipeImageSegmentationNode] Closing existing segmenter due to parameter change or initial setup.")
                    self.segmenter.close()
                except Exception as e:
                    print(f"[MediaPipeImageSegmentationNode] Error closing existing segmenter: {e}")
                finally:
                    self.segmenter = None # Ensure it's reset

            # Validate model path before attempting initialization
            if not model_path or not os.path.exists(model_path):
                # Store current params even if init fails, to avoid repeated init attempts with bad path
                self.current_model_path = model_path
                self.current_output_confidence_masks = output_confidence_masks
                raise FileNotFoundError(f"Model file path is invalid or file doesn't exist: '{model_path}'. Cannot initialize segmenter.")

            print(f"[MediaPipeImageSegmentationNode] Initializing MediaPipe Image Segmenter...")
            print(f"  Model: {model_path}")
            print(f"  Output Confidence Masks: {output_confidence_masks}")
            print(f"  Requested Delegate Mode: {delegate_mode}")

            BaseOptions = mp.tasks.BaseOptions
            ImageSegmenter = mp.tasks.vision.ImageSegmenter
            ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            try:
                # Determine delegate based on requested mode and platform compatibility
                delegate = BaseOptions.Delegate.CPU  # Default
                
                if delegate_mode.lower() == "gpu":
                    # Check if platform is Windows
                    if platform.system().lower() == 'windows':
                        raise RuntimeError("GPU delegate is not supported on Windows platforms. Please use 'cpu' delegate mode instead.")
                    
                    # Check if GPU delegate is available in MediaPipe
                    if hasattr(BaseOptions.Delegate, 'GPU'):
                        delegate = BaseOptions.Delegate.GPU
                        print("  Using GPU delegate.")
                    else:
                        raise RuntimeError("GPU delegate requested but not available in the installed MediaPipe version. Please use 'cpu' delegate mode instead.")
                else:
                    print("  Using CPU delegate.")

                print(f"  Using Delegate: {delegate.name}")

                options = ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
                    running_mode=VisionRunningMode.IMAGE,
                    output_category_mask=not output_confidence_masks,
                    output_confidence_masks=output_confidence_masks
                )

                # Create the new segmenter instance
                self.segmenter = ImageSegmenter.create_from_options(options)
                # Store the current settings used for this segmenter instance
                self.current_model_path = model_path
                self.current_output_confidence_masks = output_confidence_masks
                print(f"[MediaPipeImageSegmentationNode] MediaPipe Image Segmenter initialized successfully.")

            except Exception as e:
                # Clean up potentially partially created segmenter
                if hasattr(self, 'segmenter') and self.segmenter:
                    try:
                        self.segmenter.close()
                    except Exception as close_err:
                        print(f"[MediaPipeImageSegmentationNode] Error closing segmenter during exception handling: {close_err}")
                self.segmenter = None
                # Store params that caused the failure
                self.current_model_path = model_path
                self.current_output_confidence_masks = output_confidence_masks
                # Raise the error
                raise RuntimeError(f"Failed to initialize MediaPipe Image Segmenter: {e}")
        # else: # Optional: Log if reusing the existing segmenter
            # print("[MediaPipeImageSegmentationNode] Reusing existing segmenter instance.")

    def is_multiclass_model(self, model_name):
        """Check if the model is a multiclass segmentation model"""
        return model_name == "Selfie Multiclass"

    def create_visualization(self, category_mask_np, is_multiclass=False):
        """Create a visualization image from a category mask - optimized for maximum speed"""
        if is_multiclass:
            max_class = max(CLASS_COLORS_BGR.keys())
            lut = np.zeros((max_class + 1, 3), dtype=np.uint8)

            for class_id, color_bgr in CLASS_COLORS_BGR.items():
                lut[class_id] = [color_bgr[2], color_bgr[1], color_bgr[0]] # BGR to RGB

            return lut[category_mask_np]
        else:
            binary_mask = (category_mask_np > 0).astype(np.uint8)
            return np.stack([binary_mask * 255] * 3, axis=-1)


    # Updated function signature to accept model_info and generate_visualization
    def segment_image(self, image: torch.Tensor, model_info: dict, output_confidence_masks: bool,
                     threshold: float, generate_visualization: bool, delegate_mode: str = "cpu"):
        # Extract model path and name from the info dictionary
        model_path = model_info.get("model_path")
        model_name = model_info.get("model_name")

        if not model_path:
            raise ValueError("[MediaPipeImageSegmentationNode] Invalid model_info received (missing 'model_path')")

        # Immediately validate GPU mode on Windows before any processing
        if delegate_mode.lower() == "gpu" and platform.system().lower() == 'windows':
            raise RuntimeError("[MediaPipeImageSegmentationNode] GPU acceleration is not supported on Windows platforms. Please set delegate_mode to 'cpu'.")

        # Check if this is a multiclass model
        is_multiclass = self.is_multiclass_model(model_name)
        # Multiclass raw output is only generated when *not* outputting confidence masks
        # generate_multiclass_output = is_multiclass and not output_confidence_masks # Not strictly needed as a var

        # Ensure segmenter is initialized with the correct settings
        try:
             self._initialize_segmenter(model_path, output_confidence_masks, delegate_mode)
        except Exception as e:
            raise RuntimeError(f"[MediaPipeImageSegmentationNode] Failed to initialize segmenter: {e}")

        if self.segmenter is None:
            raise RuntimeError("[MediaPipeImageSegmentationNode] MediaPipe Image Segmenter failed to initialize")

        batch_size, height, width, channels = image.shape
        output_masks_list = []
        output_vis_list = []
        output_multiclass_list = [] # List for raw multiclass integer masks

        # --- Optimization: Pre-process the batch --- 
        # Convert the entire batch to NumPy, scale to 0-255, and convert to uint8
        # Check for NaN/inf values which can cause issues
        if torch.isnan(image).any() or torch.isinf(image).any():
             print(f"[MediaPipeImageSegmentationNode] Warning: Input batch contains NaN or Inf values. Clamping to [0, 1].")
             image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
             image = torch.clamp(image, 0.0, 1.0)
        
        # Ensure input is float32 before scaling
        if image.dtype != torch.float32:
             image = image.float() # Convert entire batch if needed
        
        # Perform scaling and uint8 conversion using NumPy for the whole batch
        # .numpy() might implicitly handle device transfer if needed, but CPU is typical here
        # Let potential errors here propagate as they indicate fundamental issues.
        image_np_uint8_batch = (image.cpu().numpy() * 255).astype(np.uint8)
        # ---------------------------------------------

        for i in range(batch_size):
            # Get the pre-processed slice from the batch
            img_rgb_uint8_slice = image_np_uint8_batch[i]
            
            # Ensure the array slice is C-contiguous (important for MediaPipe)
            img_rgb_uint8_cont = np.ascontiguousarray(img_rgb_uint8_slice)
            
            # Check shape just in case something went wrong during slicing/batching
            if img_rgb_uint8_cont.shape != (height, width, channels):
                 # Log error but continue processing other images in batch if possible
                 print(f"[MediaPipeImageSegmentationNode] Error: Unexpected image slice shape for item {i}. Expected {(height, width, channels)}, got {img_rgb_uint8_cont.shape}. Skipping this item.")
                 # Append placeholders to maintain batch size consistency for stacking later
                 output_masks_list.append(torch.zeros((height, width), dtype=torch.float32))
                 output_vis_list.append(torch.zeros((height, width, 3), dtype=torch.float32))
                 output_multiclass_list.append(torch.zeros((height, width), dtype=torch.int32))
                 continue

            # Create MediaPipe Image - Wrap specific MediaPipe call
            try:
                # SRGB assumes input is already gamma corrected (standard for display images)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_uint8_cont)
            except Exception as e:
                if batch_size == 1:
                    raise RuntimeError(f"[MediaPipeImageSegmentationNode] Error creating MediaPipe Image object: {e}")
                # Log error but continue processing other images in batch if possible
                print(f"[MediaPipeImageSegmentationNode] Error creating MediaPipe Image object for image {i}: {e}. Skipping this item.")
                # Append placeholders
                output_masks_list.append(torch.zeros((height, width), dtype=torch.float32))
                output_vis_list.append(torch.zeros((height, width, 3), dtype=torch.float32))
                output_multiclass_list.append(torch.zeros((height, width), dtype=torch.int32))
                continue

            
            segmentation_result = self.segmenter.segment(mp_image)

            # Initialize outputs for this image
            mask_np = np.zeros((height, width), dtype=np.float32) # Primary mask output (float)
            vis_img = np.zeros((height, width, 3), dtype=np.float32) # Visualization output (float)
            multiclass_mask_np = np.zeros((height, width), dtype=np.int32) # Raw multiclass output (int)
            category_mask_for_vis = None # Temp storage for visualization generation
            

            if output_confidence_masks:
                if segmentation_result.confidence_masks is not None and len(segmentation_result.confidence_masks) > 0:
                    target_mask_index = 1 if len(segmentation_result.confidence_masks) > 1 else 0
                    primary_mask = segmentation_result.confidence_masks[target_mask_index].numpy_view()
                    mask_np = np.where(primary_mask >= threshold, primary_mask, 0.0).astype(np.float32)
                    if generate_visualization:
                         vis_img_gray = (mask_np * 255).astype(np.uint8)
                         vis_img = np.stack([vis_img_gray] * 3, axis=-1)
                else:
                     # Log warning if expected masks are missing
                     print(f"[MediaPipeImageSegmentationNode] Warning: output_confidence_masks is True, but no confidence masks found for image {i}. Outputting zero mask.")
            else: # Category Masks
                if segmentation_result.category_mask is not None:
                    category_mask_np = segmentation_result.category_mask.numpy_view().astype(np.int32)
                    category_mask_for_vis = category_mask_np
                    if is_multiclass:
                        mask_np = (category_mask_np > 0).astype(np.float32)
                        multiclass_mask_np = category_mask_np
                    else:
                        mask_np = (category_mask_np > 0).astype(np.float32)
                    if generate_visualization and category_mask_for_vis is not None:
                        vis_img = self.create_visualization(category_mask_for_vis, is_multiclass)
                else:
                     # Log warning if expected mask is missing
                     print(f"[MediaPipeImageSegmentationNode] Warning: output_confidence_masks is False, but no category mask found for image {i}. Outputting zero masks.")

            
            if vis_img.dtype == np.uint8:
                vis_img = vis_img.astype(np.float32) / 255.0

            # Add outputs to lists
            output_masks_list.append(torch.from_numpy(mask_np))
            output_vis_list.append(torch.from_numpy(vis_img))
            output_multiclass_list.append(torch.from_numpy(multiclass_mask_np))

        # Handle case where the loop finished but produced no valid outputs (e.g., all images failed creation)
        if not output_masks_list:
            raise RuntimeError("[MediaPipeImageSegmentationNode] No valid masks were generated for any image in the batch. Check input images and model configuration.")


        output_mask_batch = torch.stack(output_masks_list, dim=0)
        output_vis_batch = torch.stack(output_vis_list, dim=0)
        output_multiclass_batch = torch.stack(output_multiclass_list, dim=0)
        
        # Ensure correct types before returning
        return (output_mask_batch.float(), output_vis_batch.float(), output_multiclass_batch.int())


    # Optional: Clean up the MediaPipe segmenter when the node is deleted or ComfyUI shuts down
    def __del__(self):
        if hasattr(self, 'segmenter') and self.segmenter:
            try:
                print("[MediaPipeImageSegmentationNode] Closing MediaPipe Image Segmenter...")
                self.segmenter.close()
            except Exception as e:
                print(f"[MediaPipeImageSegmentationNode] Error closing MediaPipe Image Segmenter: {e}")


class SelectMediaPipeSegmentNode:
    """
    Selects a specific segmentation mask from the raw multiclass output
    of the MediaPipeImageSegmentationNode.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiclass_segments": ("MP_INT_MASK", { # Input the raw integer mask tensor
                    "tooltip": "Raw multiclass segmentation data from MediaPipeImageSegmentationNode (only generated with Selfie Multiclass model and output_confidence_masks=False)."
                }),
                "segment_name": (list(MULTICLASS_NAMES.keys()), { # Dropdown using the defined names
                    "default": "Clothes",
                    "tooltip": "Select the specific body part or segment to extract as a mask."
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "select_segment"
    CATEGORY = "StreamPack/Segmentation"
    DESCRIPTION = "Extracts a single mask for a selected segment (e.g., Hair, Clothes) from the multiclass segmentation data provided by the MediaPipe Image Segmentation node."

    def select_segment(self, multiclass_segments: torch.Tensor, segment_name: str):
        # Validate segment_name - this is recoverable
        if segment_name not in MULTICLASS_NAMES:
            print(f"[SelectMediaPipeSegmentNode] Warning: Invalid segment_name '{segment_name}'. Available: {list(MULTICLASS_NAMES.keys())}. Returning zero mask.")
            # Attempt to get shape safely before returning zeros
            try:
                 batch, height, width = multiclass_segments.shape
                 return (torch.zeros((batch, height, width), dtype=torch.float32),)
            except Exception as e:
                 # If shape access fails, the input tensor might be invalid
                 raise ValueError(f"[SelectMediaPipeSegmentNode] Error accessing shape of input multiclass_segments tensor: {e}. Is the input valid?") from e

        target_class_id = MULTICLASS_NAMES[segment_name]
        print(f"[SelectMediaPipeSegmentNode] Extracting segment '{segment_name}' (ID: {target_class_id})...")

        # Input type check and conversion - attempt conversion but raise if it fails
        if multiclass_segments.dtype != torch.int32 and multiclass_segments.dtype != torch.int64:
             print(f"[SelectMediaPipeSegmentNode] Warning: Input multiclass_segments tensor has unexpected dtype {multiclass_segments.dtype}. Attempting conversion to int32.")
             try:
                 multiclass_segments = multiclass_segments.int()
             except Exception as e:
                 raise TypeError(f"[SelectMediaPipeSegmentNode] Failed to convert input tensor to integer type: {e}") from e

        # Create the binary mask - Let potential tensor operation errors propagate
        selected_mask = (multiclass_segments == target_class_id).float()

        return (selected_mask,)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "MediaPipeImageSegmentationNode": MediaPipeImageSegmentationNode,
    "MediaPipeModelLoaderNode": MediaPipeModelLoaderNode,
    "SelectMediaPipeSegmentNode": SelectMediaPipeSegmentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeImageSegmentationNode": "MediaPipe Image Segmentation",
    "MediaPipeModelLoaderNode": "MediaPipe Model Loader",
    "SelectMediaPipeSegmentNode": "Select MediaPipe Segment",
}
