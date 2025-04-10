#TODO: review controlnet aux structure for division of these wrapper files
from ..src.media_pipe.image_segmentation import *

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

__all__ = ["MediaPipeImageSegmentationNode", "MediaPipeModelLoaderNode", "SelectMediaPipeSegmentNode"]