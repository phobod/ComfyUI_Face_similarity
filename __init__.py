# Applied before anything else: it repairs a header that ComfyUI's own
# HF_HUB_DISABLE_TELEMETRY setting makes malformed, which otherwise stops FP8
# models from loading their kernels. See hf_kernel_fix.py.
from . import hf_kernel_fix  # noqa: F401

from .face_similarity_node import (
    NODE_CLASS_MAPPINGS as FACE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FACE_DISPLAY_MAPPINGS,
)
from .clip_embedding_node import (
    NODE_CLASS_MAPPINGS as CLIP_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CLIP_DISPLAY_MAPPINGS,
)
from .report_text_node import (
    NODE_CLASS_MAPPINGS as TEXT_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as TEXT_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {**FACE_CLASS_MAPPINGS, **CLIP_CLASS_MAPPINGS, **TEXT_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {
    **FACE_DISPLAY_MAPPINGS,
    **CLIP_DISPLAY_MAPPINGS,
    **TEXT_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
