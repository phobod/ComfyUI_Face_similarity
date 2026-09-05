"""Nodes for getting CLIP vision embeddings out of ComfyUI and onto disk.

ComfyUI can produce a CLIP_VISION_OUTPUT but has no way to save one, so an
embedding cannot leave a workflow. These nodes close that gap for two jobs:

* **SaveClipEmbedding** writes the image embedding as a ``.npy`` file, so a
  batch of images becomes a set of vectors that some other process can compare
  pairwise — near-duplicate detection over a whole collection, which cannot be
  done one image at a time inside a workflow.
* **AestheticScore** runs the LAION aesthetic head and returns a number, so the
  score is available without the caller needing torch at all.

Both take ``image_embeds`` — the projected, 768-dimensional vector. Not
``penultimate_hidden_states``, which is what IPAdapter consumes and what the
aesthetic head would silently mis-score.
"""

import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn

#: The LAION aesthetic head, trained on OpenAI CLIP ViT-L/14 embeddings. The
#: "l14" in the name is load-bearing: the first layer takes 768 inputs, so a
#: ViT-H (1024) or SigLIP (1152) embedding will not fit it.
AESTHETIC_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac+logos+ava1-l14-linearMSE.pth"
)
AESTHETIC_WEIGHTS_NAME = "sac+logos+ava1-l14-linearMSE.pth"

_aesthetic_head = None


class AestheticMLP(nn.Module):
    """The 768 -> 1 stack the LAION predictor's weights were trained as."""

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def get_aesthetic_head(device) -> AestheticMLP:
    """Load and cache the aesthetic head, downloading the weights once."""
    global _aesthetic_head
    if _aesthetic_head is None:
        import folder_paths

        directory = os.path.join(folder_paths.models_dir, "aesthetic")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, AESTHETIC_WEIGHTS_NAME)
        if not os.path.exists(path):
            print(f"[AestheticScore] downloading weights to {path}")
            urllib.request.urlretrieve(AESTHETIC_WEIGHTS_URL, path)

        model = AestheticMLP()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _aesthetic_head = model
    return _aesthetic_head.to(device)


def get_image_embeds(clip_vision_output, normalize: bool) -> np.ndarray:
    """The projected image embedding as [batch, 768] float32.

    Normalising to unit length is what both consumers want: cosine similarity
    reduces to a dot product, and the aesthetic head was trained on normalised
    input. It is a flag rather than a given so a caller can keep the raw
    magnitudes if they ever mean something.
    """
    embeds = clip_vision_output.image_embeds
    if normalize:
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.detach().cpu().float().numpy()


class SaveClipEmbedding:
    """Write a CLIP vision embedding to a .npy file.

    ``path`` is a full output path **without** an extension, matching the
    convention PathSaveImageRGB uses, and a relative one resolves against
    ComfyUI's own working directory rather than the caller's — so callers driving
    this over the API should send absolute paths.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "path": ("STRING", {"default": "embedding", "multiline": False}),
                "normalize": ("BOOLEAN", {"default": True}),
                "create_dirs": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("path", "dimensions")
    FUNCTION = "save_embedding"
    CATEGORY = "CLIP Vision"
    OUTPUT_NODE = True

    def save_embedding(self, clip_vision_output, path: str, normalize: bool, create_dirs: bool):
        embeds = get_image_embeds(clip_vision_output, normalize)

        target = path if path.lower().endswith(".npy") else path + ".npy"
        directory = os.path.dirname(os.path.abspath(target))
        if create_dirs:
            os.makedirs(directory, exist_ok=True)
        elif not os.path.isdir(directory):
            msg = f"directory does not exist: {directory}"
            print(f"[SaveClipEmbedding] {msg}")
            return {"ui": {"text": [msg]}, "result": ("", 0)}

        np.save(target, embeds)
        preview = f"{embeds.shape[0]}x{embeds.shape[1]} -> {os.path.basename(target)}"
        print(f"[SaveClipEmbedding] saved {preview}")
        return {"ui": {"text": [preview]}, "result": (target, int(embeds.shape[-1]))}


class AestheticScore:
    """LAION aesthetic score for a CLIP ViT-L/14 embedding, roughly 1-10.

    Returns the batch mean as the FLOAT output; per-image values go in the
    string, which is what a batch of more than one is usually wanted for.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "scores")
    FUNCTION = "score"
    CATEGORY = "CLIP Vision"
    OUTPUT_NODE = True

    def score(self, clip_vision_output):
        embeds = clip_vision_output.image_embeds
        if embeds.shape[-1] != 768:
            msg = (
                f"embedding is {embeds.shape[-1]}-dimensional; the LAION head needs 768 "
                "(OpenAI CLIP ViT-L/14). A ViT-H or SigLIP vision model will not fit it."
            )
            print(f"[AestheticScore] {msg}")
            return {"ui": {"text": [msg]}, "result": (0.0, msg)}

        device = embeds.device
        head = get_aesthetic_head(device)
        normalised = embeds / embeds.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            values = head(normalised.to(device).float()).squeeze(-1).cpu().tolist()

        mean = sum(values) / len(values)
        detail = ", ".join(f"{v:.3f}" for v in values)
        print(f"[AestheticScore] {detail}")
        return {"ui": {"text": [f"{mean:.3f}"]}, "result": (float(mean), detail)}


NODE_CLASS_MAPPINGS = {
    "SaveClipEmbedding": SaveClipEmbedding,
    "AestheticScore": AestheticScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveClipEmbedding": "Save CLIP Embedding (.npy)",
    "AestheticScore": "Aesthetic Score (LAION)",
}
