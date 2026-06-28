"""RunPod serverless handler for WhisperX + pyannote diarization.

job_input fields:
  audio_base64 : str   -- base64-encoded WAV
  language     : str   -- ISO-639-1 (default "ar")
  model        : str   -- HF repo id
  min_speakers : int   -- optional
  max_speakers : int   -- optional

Returns:
  segments : [{start, end, text, speaker}]
  speakers : ["SPEAKER_00", "SPEAKER_01", ...]   (sorted)
  model    : <repo_id>

WIRE COMPATIBLE with docs/sample-remote-worker/handler.py — the same JSON
shape that meetily's RemoteProvider posts/receives (PR #528, #532).
"""
import os, base64, tempfile, traceback
import runpod
import torch

_MODEL = _ALIGN = _META = _DIARIZE = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load(model_id: str) -> None:
    global _MODEL, _ALIGN, _META, _DIARIZE
    if _MODEL is not None:
        return
    import whisperx
    compute_type = "float16" if _DEVICE == "cuda" else "int8"
    hf_token = os.environ.get("HF_TOKEN")
    _MODEL = whisperx.load_model(model_id, device=_DEVICE, compute_type=compute_type)
    _ALIGN, _META = whisperx.load_align_model(language_code="ar", device=_DEVICE)
    _DIARIZE = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=_DEVICE)

def handler(event):
    job_input = event.get("input", {})
    audio_b64 = job_input.get("audio_base64")
    if not audio_b64:
        return {"error": "audio_base64 is required"}
    language = job_input.get("language") or os.environ.get("DEFAULT_LANGUAGE", "ar")
    model_id = job_input.get("model") or os.environ.get(
        "DEFAULT_MODEL",
        "Mano200600/faster-whisper-large-v2-ar-codeswitching",
    )
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(base64.b64decode(audio_b64))
            wav_path = tf.name

        import whisperx
        _load(model_id)

        audio = whisperx.load_audio(wav_path)
        result = _MODEL.transcribe(audio, language=language)
        result = whisperx.align(result["segments"], _ALIGN, _META, audio, device=_DEVICE)

        diar_kwargs = {}
        if (ms := job_input.get("min_speakers")) is not None:
            diar_kwargs["min_speakers"] = ms
        if (mx := job_input.get("max_speakers")) is not None:
            diar_kwargs["max_speakers"] = mx
        diarize_segments = _DIARIZE(audio, **diar_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        out = []
        speakers = set()
        for seg in result.get("segments", []):
            spk = seg.get("speaker") or "UNKNOWN"
            speakers.add(spk)
            out.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
                "speaker": spk,
            })
        return {
            "segments": out,
            "speakers": sorted(speakers),
            "model": model_id,
        }
    except Exception as exc:
        return {"error": str(exc), "trace": traceback.format_exc()}
    finally:
        try:
            if wav_path:
                os.unlink(wav_path)
        except OSError:
            pass

runpod.serverless.start({"handler": handler})
