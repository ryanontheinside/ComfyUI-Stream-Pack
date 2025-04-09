#TODO: review controlnet aux structure for division of these wrapper files
from ..src.media_pipe.image_segmentation import *

NODE_CLASS_MAPPINGS = {

    "MediaPipeImageSegmentationNode": MediaPipeImageSegmentationNode,
    "MediaPipeModelLoaderNode": MediaPipeModelLoaderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeImageSegmentationNode": "MediaPipe Image Segmentation",
    "MediaPipeModelLoaderNode": "MediaPipe Model Loader"
}

__all__ = ["MediaPipeImageSegmentationNode", "MediaPipeModelLoaderNode"]