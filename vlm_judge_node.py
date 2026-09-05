"""A vision-language model that answers exactly the prompt it was given.

Written for QA work, where the prompt is the instrument. General captioning nodes
append a preset system prompt to whatever you type, which is fine for captioning
and wrong for judging: the judgement has to be reproducible from the prompt
alone, and a template glued on the end quietly competes with it.

What this does differently:

* **The prompt is sent verbatim.** Nothing is prepended or appended.
* **The model is a plain repo id**, so it needs no registry file inside a
  third-party package — nothing to lose when that package updates.
* **Deterministic by default.** ``temperature = 0`` means greedy decoding, so the
  same image and prompt give the same answer across runs. A judge that wanders
  cannot be calibrated.
* **Loaded straight onto the GPU**, not to CPU and then moved.
* **Unloadable.** ``keep_loaded = false`` drops the model at the end of the run,
  which ComfyUI's own ``/free`` cannot do for a model a custom node is holding.
* **It is an output node**, so the answer reaches ``/history`` and any caller
  driving ComfyUI over the API can read it without a second node.

FP8 repositories work because transformers reads the quantization config from the
repo itself; they also need the `kernels` package (see hf_kernel_fix.py, which
repairs a header that stops it from loading under ComfyUI).
"""

import gc
import json

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"

# One model at a time, keyed by what would require a reload.
_LOADED: dict = {"key": None, "model": None, "processor": None}


def _unload() -> None:
    _LOADED.update(key=None, model=None, processor=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(model_id: str, device: str):
    """Load the model, reusing the resident one when nothing relevant changed."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    key = (model_id, device)
    if _LOADED["key"] == key and _LOADED["model"] is not None:
        return _LOADED["model"], _LOADED["processor"]

    _unload()
    print(f"[VLMJudge] loading {model_id} onto {device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map=device,
        dtype="auto",  # an FP8 repo carries its own; "auto" respects it
        attn_implementation="sdpa",  # the only backend FP8 weights work with here
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).eval()

    _LOADED.update(key=key, model=model, processor=processor)
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[VLMJudge] resident: {used:.1f} of {total:.1f} GB")
    return model, processor


def _to_pil(tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


class VLMJudge:
    """Ask a vision-language model one question about one image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model_id": ("STRING", {"default": DEFAULT_MODEL, "multiline": False}),
                "max_tokens": ("INT", {"default": 384, "min": 16, "max": 4096}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.5, "step": 0.05,
                     "tooltip": "0 means greedy decoding: same image and prompt, same answer."},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1,
                                 "tooltip": "Only matters when temperature is above 0."}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
                "require_json": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("response", "is_json")
    FUNCTION = "judge"
    CATEGORY = "QA"
    OUTPUT_NODE = True

    @torch.no_grad()
    def judge(
        self,
        image,
        prompt: str,
        model_id: str = DEFAULT_MODEL,
        max_tokens: int = 384,
        temperature: float = 0.0,
        seed: int = 0,
        keep_loaded: bool = True,
        require_json: bool = False,
        device: str = "cuda",
    ):
        if not prompt.strip():
            message = "no prompt given"
            return {"ui": {"text": [message]}, "result": (message, False)}

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model, processor = _load(model_id, device)

        conversation = [{
            "role": "user",
            "content": [{"type": "image", "image": _to_pil(image)},
                        {"type": "text", "text": prompt}],
        }]
        chat = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=chat, images=[_to_pil(image)], return_tensors="pt")
        target = next(model.parameters()).device
        inputs = {k: (v.to(target) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        kwargs = {"max_new_tokens": max_tokens}
        if temperature > 0:
            torch.manual_seed(seed)
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            kwargs["do_sample"] = False

        generated = model.generate(**inputs, **kwargs)
        # Keep only what the model added; the prompt is echoed back in the ids.
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        text = processor.decode(new_tokens, skip_special_tokens=True).strip()

        is_json = True
        if require_json:
            try:
                json.loads(_strip_fence(text))
            except (ValueError, TypeError):
                is_json = False
                print(f"[VLMJudge] answer is not JSON: {text[:200]}")

        if not keep_loaded:
            _unload()

        return {"ui": {"text": [text]}, "result": (text, is_json)}


class VLMUnload:
    """Drop the resident model. ComfyUI's own /free cannot reach it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"any": ("*",)}}

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "QA"
    OUTPUT_NODE = True

    def unload(self, any=None):
        was = _LOADED["key"]
        _unload()
        message = f"unloaded {was[0]}" if was else "nothing was loaded"
        print(f"[VLMJudge] {message}")
        return {"ui": {"text": [message]}}


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()


NODE_CLASS_MAPPINGS = {"VLMJudge": VLMJudge, "VLMUnload": VLMUnload}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMJudge": "VLM Judge (one image, one question)",
    "VLMUnload": "VLM Unload",
}
