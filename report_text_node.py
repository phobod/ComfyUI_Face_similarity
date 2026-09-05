"""Get a string out of a workflow.

Many nodes produce a STRING but are not output nodes, so their result dies inside
the graph — a caption, a judgement, a VLM's verdict never reaches the caller. An
output node's ``ui`` payload, on the other hand, lands in ``/history/<prompt_id>``
keyed by the node's own id, which is how anything driving ComfyUI over the API
reads a result back without a file to carry it.

``ReportText`` is that adapter and nothing more: a string goes in, the same string
comes out, and it appears in the history payload on the way past.
"""

import json


class ReportText:
    """Report a STRING in the /history payload, and pass it through unchanged.

    ``label`` is written alongside the text so a graph with several of these
    stays readable in the ComfyUI UI; it does not change what is reported.

    ``require_json`` is for callers that asked the model for structured output.
    It does not repair anything — it reports whether the string parsed, so a
    batch run can count how often the model ignored the format instead of
    discovering it later while reading results.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "label": ("STRING", {"default": "", "multiline": False}),
                "require_json": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("text", "is_json")
    FUNCTION = "report"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def report(self, text, label: str = "", require_json: bool = False):
        # A STRING input can arrive as a one-item list when it comes from a node
        # that batches; take the first rather than reporting "['...']".
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        text = "" if text is None else str(text)

        is_json = True
        if require_json:
            try:
                json.loads(_strip_code_fence(text))
            except (ValueError, TypeError):
                is_json = False
                print(f"[ReportText] {label or 'text'} is not valid JSON: {text[:200]}")

        return {"ui": {"text": [text]}, "result": (text, is_json)}


def _strip_code_fence(text: str) -> str:
    """Models wrap JSON in ``` fences more often than not."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()


NODE_CLASS_MAPPINGS = {"ReportText": ReportText}
NODE_DISPLAY_NAME_MAPPINGS = {"ReportText": "Report Text (to /history)"}
