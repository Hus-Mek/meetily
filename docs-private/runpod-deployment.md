# RunPod Deployment Runbook (fork-only, never PR'd)

> Lives in our private fork. Not part of meetily. **Never** submitted to upstream.

## TL;DR

Meetily's `RemoteProvider` (PR #528 + #532) POSTs WAV bytes to any HTTPS endpoint. We host that endpoint on RunPod serverless running on RTX 3090, with WhisperX for Arabic/code-switching ASR and pyannote for diarization.

## Endpoint config

1. Build locally & smoke test:
   ```bash
   cd runpod-worker
   docker build -t meetily-runpod-whisperx:latest .
   docker run --rm --gpus all -e HF_TOKEN=*** \
     -v "$PWD:/app" meetily-runpod-whisperx:latest \
     python3.11 scripts/test_local.py /test_audio/test.wav
   ```
2. Push image:
   ```bash
   docker tag meetily-runpod-whisperx:latest <registry>/<image>:v0.1.0
   docker push <registry>/<image>:v0.1.0
   ```
3. RunPod console → Serverless → New Endpoint
   - GPU: RTX 3090 (24 GB)
   - Container disk: 10 GB
   - Max workers: 2 initially (bump to 5–10 once we know steady-state)
   - Idle timeout: 5 s
   - Env: `HF_TOKEN`, `DEFAULT_MODEL`, `DEFAULT_LANGUAGE`
4. Smoke-test the live endpoint:
   ```bash
   curl -X POST https://api.runpod.ai/v2/<endpoint_id>/runsync \
     -H "Authorization: Bearer *** \
     -H "Content-Type: application/json" \
     -d "$(jq -n --rawfile wav /test_audio/test.wav '{input: {audio_base64: $wav|@base64}, language: "ar"}')"
   ```

## Meetily settings (manual UI)

- Settings → Transcription → Provider: **Remote HTTPS**
- Endpoint URL: `https://api.runpod.ai/v2/<endpoint_id>/runsync`
- Bearer token: RunPod API key
- Model: `Mano200600/faster-whisper-large-v2-ar-codeswitching`
- Default language: `ar`
- Min/Max speakers: optional, leave blank for auto

## Costs (RTX 3090, ~$0.36/hr idle + per-second compute)

- ~$0.024 per 1-hour meeting
- Cost dominated by worker spin-up; idle timeout = 5 s keeps it short
- For batch retranscribe jobs, push multiple files per run

## Operational notes

- The worker exposes the **same wire contract** as `docs/sample-remote-worker/handler.py`, so any change to the contract needs migration on BOTH this worker and any self-hosted worker downstream
- HF_TOKEN is required for pyannote-audio model access. Lost/rotated → worker cold-start fails 401
- WhisperX 3.4 is pinned; bumping to 4.x needs re-evaluating alignment stage format

## What this is NOT

- This directory is a deployment config, not part of meetily
- Upstream maintainers don't see it
- No settings table column hardcoded to RunPod — generic `endpoint_url` + `bearer_token` only
