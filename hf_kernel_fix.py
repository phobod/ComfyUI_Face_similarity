"""Repair the User-Agent that `kernels` builds when HF telemetry is disabled.

ComfyUI's ``main.py`` sets ``HF_HUB_DISABLE_TELEMETRY=1`` unconditionally at
startup. With telemetry off, ``kernels.utils._get_hf_api`` builds its
``user_agent`` as an empty string rather than ``None``, and
``huggingface_hub`` then appends that empty field to the header:

    user_agent=''    ->  'kernels/0.15.2; hf_hub/1.22.0; python/3.12.3; '
    user_agent=None  ->  'kernels/0.15.2; hf_hub/1.22.0; python/3.12.3'

httpx rejects the trailing ``"; "`` as an illegal header value. The failure
surfaces far from its cause: the publisher-trust lookup catches every exception
and reports "could not verify publisher trust status", which reads like a
permissions problem and is not one. The visible symptom is that an FP8 model
cannot load its finegrained-fp8 kernel.

The fix is one field: send no user-agent instead of an empty one. Telemetry
stays off — nothing extra is sent, the malformed empty field is simply dropped.

Patching here rather than in ComfyUI's ``main.py`` or in site-packages means it
survives an update to either. It is a no-op when ``kernels`` is absent, or once
the upstream bug is fixed.
"""


def apply() -> str | None:
    """Patch the user-agent builder. Returns a message when it changed something."""
    try:
        import kernels.utils as kernel_utils
    except ImportError:
        return None

    original = getattr(kernel_utils, "_get_hf_api", None)
    if original is None or getattr(original, "_ua_patched", False):
        return None

    def _get_hf_api(user_agent=None):
        api = original(user_agent)
        # Only the empty string is the bug; a populated agent is left alone, so
        # this stops doing anything the moment upstream stops producing one.
        if getattr(api, "user_agent", None) == "":
            api.user_agent = None
        return api

    _get_hf_api._ua_patched = True
    kernel_utils._get_hf_api = _get_hf_api
    return "[hf_kernel_fix] patched kernels user-agent (empty -> None)"


_message = apply()
if _message:
    print(_message)

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
