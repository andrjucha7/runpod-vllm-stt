import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from faster_whisper import WhisperModel
import uvicorn

APP_HOST = os.getenv("STT_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("STT_PORT", "8001"))

# Model settings via env so you can tune without rebuilding image
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium/large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu or cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # cpu: int8; cuda: float16/int8_float16

app = FastAPI()

model = WhisperModel(
    WHISPER_MODEL,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

@app.get("/health")
def health():
    return {"ok": True, "model": WHISPER_MODEL, "device": WHISPER_DEVICE, "compute_type": WHISPER_COMPUTE_TYPE}

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    prompt: str = Form(default=None),
    temperature: float = Form(default=0.0),
):
    # Write upload to a temp file (ffmpeg is installed in image for decoding if needed)
    suffix = os.path.splitext(file.filename or "")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    segments, info = model.transcribe(
        tmp_path,
        language=language,
        initial_prompt=prompt,
        temperature=temperature,
        vad_filter=True,
    )

    text = "".join([seg.text for seg in segments]).strip()
    return {"text": text, "language": info.language, "duration": info.duration}

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
