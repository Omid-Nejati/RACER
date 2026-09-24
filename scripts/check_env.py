import sys

print("python:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    raise SystemExit(f"torch import failed: {e}")

try:
    import transformers
    print("transformers:", transformers.__version__)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print("Qwen2_5_VLForConditionalGeneration: OK")
except Exception as e:
    raise SystemExit(f"transformers/Qwen2.5-VL import failed: {e}")

try:
    import huggingface_hub
    print("huggingface_hub:", huggingface_hub.__version__)
except Exception as e:
    raise SystemExit(f"huggingface_hub import failed: {e}")

try:
    import generate_chair
    import ott_qwen
    import saliency_qwen
    print("MCoT + OTT + Saliency imports: OK")
except Exception as e:
    raise SystemExit(f"integration import failed: {e}")
