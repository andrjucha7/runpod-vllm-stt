import os
import tempfile
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

HOST = os.getenv("STT_HOST", "0.0.0.0")
PORT = int(os.getenv("STT_PORT", "8000"))
# Defaults to the CTranslate2-converted Swiss German model baked in at
# /models/stt by Dockerfile.stt. faster-whisper only loads CTranslate2 weights,
# so an HF transformers repo id here (e.g. openai/whisper-large-v3) will
# download and then fail to load — it must be converted first.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "/models/stt")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
# Whisper has no Swiss German ("gsw") token; gsw fine-tunes decode as German.
# Set empty to let Whisper autodetect instead.
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "de") or None

model_loaded = False
model = None

@app.on_event("startup")
async def startup_event():
    global model_loaded, model
    try:
        from faster_whisper import WhisperModel
        start = time.time()
        model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        model_loaded = True
        print(
            f"READY stt host={HOST} port={PORT} model={WHISPER_MODEL} "
            f"device={WHISPER_DEVICE} compute_type={WHISPER_COMPUTE_TYPE} "
            f"language={WHISPER_LANGUAGE} load_seconds={time.time()-start:.2f}",
            flush=True
        )
    except Exception as e:
        print(f"FATAL stt startup failed: {e}", flush=True)
        raise

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    if model_loaded:
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "loading"})

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form(default="")
):
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    started = time.time()
    # Never build the path from file.filename: it is client-controlled and a
    # value like "../app/stt_server.py" would escape /tmp.
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, language=WHISPER_LANGUAGE)
        text = "".join(segment.text for segment in segments).strip()
        latency_ms = int((time.time() - started) * 1000)
        print(
            f"REQ route=/v1/audio/transcriptions latency_ms={latency_ms} "
            f"model={WHISPER_MODEL} status=200",
            flush=True
        )
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
        }
    except Exception as e:
        latency_ms = int((time.time() - started) * 1000)
        print(
            f"REQ route=/v1/audio/transcriptions latency_ms={latency_ms} "
            f"model={WHISPER_MODEL} status=500 error={e}",
            flush=True
        )
        raise
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)