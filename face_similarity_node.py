import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis

AVAILABLE_MODELS = ["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"]
DETECTION_SIZES = ["640", "320", "160"]

_face_apps: dict = {}


def get_face_app(model_name: str, det_size: int) -> FaceAnalysis:
    """Load and cache a FaceAnalysis app per (model_name, det_size) pair."""
    key = (model_name, det_size)
    if key not in _face_apps:
        app = FaceAnalysis(
            name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(det_size, det_size))
        _face_apps[key] = app
    return _face_apps[key]


def tensor_to_cv2(tensor_image) -> np.ndarray:
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] float32 0-1 to OpenCV BGR uint8."""
    img = tensor_image[0].numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_largest_face_embedding(cv2_image: np.ndarray, app: FaceAnalysis):
    """Return the normalized embedding of the largest detected face, or None."""
    faces = app.get(cv2_image)
    if not faces:
        return None
    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return largest.normed_embedding


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings. Range: -1 to 1."""
    return float(np.dot(emb1, emb2))


class FaceSimilarityNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "model": (AVAILABLE_MODELS, {"default": "buffalo_l"}),
                "detection_size": (DETECTION_SIZES, {"default": "640"}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("image", "similarity_score", "verdict", "is_same_person")
    FUNCTION = "compare_faces"
    CATEGORY = "Face Analysis"
    OUTPUT_NODE = True

    def compare_faces(
        self,
        reference_image,
        generated_image,
        model: str,
        detection_size: str,
        threshold: float,
    ):
        app = get_face_app(model, int(detection_size))

        ref_emb = get_largest_face_embedding(tensor_to_cv2(reference_image), app)
        gen_emb = get_largest_face_embedding(tensor_to_cv2(generated_image), app)

        if ref_emb is None:
            msg = "No face detected in reference image."
            return {"ui": {"text": [msg]}, "result": (generated_image, 0.0, msg, False)}
        if gen_emb is None:
            msg = "No face detected in generated image."
            return {"ui": {"text": [msg]}, "result": (generated_image, 0.0, msg, False)}

        score = cosine_similarity(ref_emb, gen_emb)
        is_same = score >= threshold

        verdict = (
            f"Same person — score: {score:.3f} (threshold: {threshold})"
            if is_same
            else f"Identity drift detected — score: {score:.3f} (threshold: {threshold})"
        )

        preview = f"score: {score:.3f}  {'✓' if is_same else '✗'}"
        return {"ui": {"text": [preview]}, "result": (generated_image, score, verdict, is_same)}


class SaveImageIfNode:
    """
    Saves an image only when the boolean flag is True.
    Pass-throughs both image and flag for easy chaining.
    """

    def __init__(self):
        import folder_paths
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "condition": ("BOOLEAN",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "condition")
    FUNCTION = "save_if"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def save_if(self, image, condition: bool, filename_prefix: str):
        if not condition:
            print("[SaveImageIf] Condition is False — image not saved.")
            return {"ui": {"text": ["Skipped."]}, "result": (image, False)}

        from PIL import Image as PILImage
        import folder_paths

        full_output_folder, filename, counter, subfolder, filename_prefix_ = \
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                image.shape[2],  # width
                image.shape[1],  # height
            )

        saved = []
        for batch_idx, img_tensor in enumerate(image):
            arr = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            pil_img = PILImage.fromarray(arr)
            fname = f"{filename}_{counter + batch_idx:05}.png"
            pil_img.save(os.path.join(full_output_folder, fname), compress_level=4)
            saved.append({"filename": fname, "subfolder": subfolder, "type": self.type})
            print(f"[SaveImageIf] Saved: {fname}")

        return {"ui": {"images": saved}, "result": (image, True)}

NODE_CLASS_MAPPINGS = {
    "FaceSimilarityNode": FaceSimilarityNode,
    "SaveImageIfNode": SaveImageIfNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSimilarityNode": "Face Similarity (InsightFace)",
    "SaveImageIfNode": "Save Image If",
}