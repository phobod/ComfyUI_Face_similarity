# ComfyUI Face Similarity (InsightFace)

A custom ComfyUI node that compares face identity between two images using [InsightFace](https://github.com/deepinsight/insightface) embeddings.  
Useful for verifying that a generated image preserves the identity of a reference person.

---

## How it works

InsightFace extracts a 512-dimensional **face embedding** from each image.  
The node computes the **cosine similarity** between the two embeddings and returns a score in the range `[-1, 1]`, where `1` means identical and `0` means unrelated.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/phobod/ComfyUI_Face_similarity.git
cd ComfyUI_Face_similarity
pip install -r requirements.txt
```

> **CPU users:** edit `requirements.txt` and swap `onnxruntime-gpu` for `onnxruntime` before running pip install.

The InsightFace model weights are downloaded automatically on first use to `~/.insightface/models/`.

---

## Node inputs

| Input | Type | Description |
|---|---|---|
| `reference_image` | IMAGE | Source photo of the person |
| `generated_image` | IMAGE | Output image to verify |
| `model` | dropdown | InsightFace model to use (see below) |
| `detection_size` | dropdown | Face detector input resolution: `640`, `320`, `160` |
| `threshold` | float 0–1 | Minimum score to consider the same person |

---

## Node outputs

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Pass-through of `generated_image` |
| `similarity_score` | FLOAT | Cosine similarity score |
| `verdict` | STRING | Human-readable result string |
| `is_same_person` | BOOLEAN | `True` if score ≥ threshold |

---

## Available models

| Model | Size | Notes |
|---|---|---|
| `buffalo_l` | ~500 MB | Most accurate, recommended default |
| `buffalo_m` | ~200 MB | Good balance of speed and accuracy |
| `buffalo_s` | ~100 MB | Faster, slightly less accurate |
| `buffalo_sc` | ~100 MB | Small, single-channel variant |
| `antelopev2` | ~300 MB | High quality alternative |

All models are downloaded automatically from the InsightFace model zoo.

---

## Score reference

| Score | Interpretation |
|---|---|
| `> 0.6` | Confidently the same person |
| `0.4 – 0.6` | Likely the same, worth checking |
| `< 0.4` | Identity has drifted |

A threshold of **0.6** is a good starting point. Adjust based on your use case.

---

## Example workflow

```
[Reference Image] ──┐
                    ├──► [Face Similarity] ──► similarity_score
[Generated Image] ──┘         │               verdict
                          is_same_person ──► (route / filter / re-sample)
```

Connect `is_same_person` (BOOLEAN) to a routing node such as `ImpactConditional` to automatically re-generate if identity drifts.

---

## Requirements

- Python 3.9+
- ComfyUI (any recent version)
- `insightface>=0.7.3`
- `onnxruntime>=1.16.0` (CPU) or `onnxruntime-gpu>=1.16.0` (CUDA)
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`

## License

This repository's code is licensed under the **MIT License**.

> **Note on model licenses:**  
> The InsightFace models downloaded at runtime (`buffalo_*`, `antelopev2`)  
> are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)  
> and are **not permitted for commercial use**.  
> For commercial projects, verify licensing with the  
> [InsightFace authors](https://github.com/deepinsight/insightface).