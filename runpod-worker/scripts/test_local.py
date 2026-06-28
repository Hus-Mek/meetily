"""Local smoke test — run after building the image, before pushing to RunPod.

Usage:
  python scripts/test_local.py <path-to-wav>
"""
import base64, json, sys
from handler import handler

if len(sys.argv) < 2:
    print("usage: test_local.py <path-to-wav>")
    sys.exit(2)

wav_path = sys.argv[1]
with open(wav_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

event = {"input": {"audio_base64": audio_b64, "language": "ar"}}
result = handler(event)
print(json.dumps(result, indent=2)[:4000])
